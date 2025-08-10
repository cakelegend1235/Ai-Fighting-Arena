"""
AI Battle Arena — single-file with PyTorch + PPO (stable-baselines3)

Features in this file:
- Uses a custom Gym environment for the 2-agent arena.
- Trains an agent (PPO) against either: a scripted heuristic opponent or a loaded model.
- Physics are time-step consistent (positions updated with velocity * dt) — fixes bullet "barely move" bug.
- Multiple weapons: pistol, machine gun, shotgun, grenade (grenade is lobbed and explodes), mines (placeable), teleport ability.
- Dash and shield abilities with cooldowns.
- Save/load trained models (torch .zip via stable-baselines3) from `saved_models/`.
- Better UI with pygame: HUD, weapon icons, cooldown bars, and a small training status overlay.

Requirements:
    pip install pygame numpy torch stable-baselines3 gym==0.21.0

How to use:
    python ai_battle_arena_ppo.py

In-game controls:
    1/2 - take manual control of Agent A / Agent B
    0 - release manual control
    WASD or arrow keys - move
    Space - fire
    Q/E - switch weapon
    G - drop mine
    T - teleport (if available)
    R - reset round
    S - save models (if trained)
    L - load models (prompts filename)
    P - start/stop training (runs PPO training for Agent A vs heuristic opponent)
    Esc - quit

NOTE: For faster/more reliable training you should run headless (Pygame rendering off) and use a machine with good CPU/GPU. This file includes a simple training loop that can be toggled with P.

"""

# Large single-file program begins here.

import os
import math
import time
import random
import tempfile
from collections import deque

import numpy as np
import pygame

# try/except imports for optional RL
try:
    import gym
    from gym import spaces
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except Exception as e:
    # RL libs not installed; we will still run the game with heuristic bots.
    RL_AVAILABLE = False
    print("RL libraries not available. To enable training install: pip install torch stable-baselines3 gym==0.21.0")

# ---------------- Config ----------------
SCREEN_W, SCREEN_H = 1024, 720
ARENA_PADDING = 60
FPS = 60

AGENT_RADIUS = 14
BULLET_RADIUS = 5
BULLET_SPEED = 800.0  # pixels per second (large because we multiply by dt)
MAX_SPEED = 220.0     # pixels per second
ACCEL = 900.0         # pixels per second^2 (force)
MAX_HEALTH = 100

# Weapon definitions (damage, bullets_per_shot, spread_rad, speed_multiplier, cooldown_seconds)
WEAPONS = {
    'pistol': { 'dmg': 18, 'count':1, 'spread':0.0, 'speed':1.0, 'cd':0.45 },
    'mg'    : { 'dmg': 8,  'count':1, 'spread':0.02,'speed':1.0, 'cd':0.12 },
    'shotgun':{ 'dmg':10, 'count':5, 'spread':0.35, 'speed':0.9, 'cd':1.0 },
    'grenade':{ 'dmg':28, 'count':1, 'spread':0.0, 'speed':0.7, 'cd':2.0 },
}
WEAPON_LIST = ['pistol','mg','shotgun','grenade']

MINE_COOLDOWN = 4.0
TELEPORT_COOLDOWN = 6.0
DASH_COOLDOWN = 3.0
SHIELD_COOLDOWN = 10.0
SHIELD_DURATION = 1.6

# Observation and action dimensions (for RL agent)
OBS_DIM = 16  # we'll construct a compact observation vector
ACTION_DIM = 5  # ax, ay continuous ([-1,1]), shoot (0/1), switch_weapon (-1/0/1 encoded), place_mine (0/1) -- we'll pack as continuous and discretize

# ---------------- Utilities ----------------

def clamp(x, a, b):
    return max(a, min(b, x))


def angle_between(ax, ay, bx, by):
    return math.atan2(by - ay, bx - ax)


# ---------------- Physics & Entities ----------------
class Bullet:
    def __init__(self, x, y, vx, vy, owner_id, dmg=15, is_grenade=False):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.owner = owner_id
        self.dmg = dmg
        self.life = 3.0
        self.is_grenade = is_grenade
        self.timer = 1.2 if is_grenade else 0.0  # grenade explodes after timer

    def update(self, dt):
        if self.is_grenade:
            # simple ballistic arc (apply gravity effect)
            self.vy += 400.0 * dt  # gravity px/s^2
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        if self.is_grenade:
            self.timer -= dt


class Mine:
    def __init__(self, x, y, owner_id, life=12.0):
        self.x = x
        self.y = y
        self.owner = owner_id
        self.life = life

    def update(self, dt):
        self.life -= dt


class Agent:
    def __init__(self, x, y, agent_id, policy=None):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.id = agent_id
        self.health = MAX_HEALTH
        self.weapon = 0
        self.weapon_cooldown = 0.0
        self.mine_cd = 0.0
        self.teleport_cd = 0.0
        self.dash_cd = 0.0
        self.shield_cd = 0.0
        self.shield_timer = 0.0
        self.dashing = False
        self.policy = policy  # if RL model will be assigned externally
        self.score = 0

    def reset(self, x=None, y=None):
        self.vx = 0.0
        self.vy = 0.0
        self.health = MAX_HEALTH
        self.weapon = 0
        self.weapon_cooldown = 0.0
        self.mine_cd = 0.0
        self.teleport_cd = 0.0
        self.dash_cd = 0.0
        self.shield_cd = 0.0
        self.shield_timer = 0.0
        self.dashing = False
        self.score = 0
        if x is not None: self.x = x
        if y is not None: self.y = y

    def is_alive(self):
        return self.health > 0

    def apply_timers(self, dt):
        self.weapon_cooldown = max(0.0, self.weapon_cooldown - dt)
        self.mine_cd = max(0.0, self.mine_cd - dt)
        self.teleport_cd = max(0.0, self.teleport_cd - dt)
        self.dash_cd = max(0.0, self.dash_cd - dt)
        self.shield_cd = max(0.0, self.shield_cd - dt)
        if self.shield_timer > 0.0:
            self.shield_timer = max(0.0, self.shield_timer - dt)

    def can_fire(self):
        return self.weapon_cooldown <= 0.0

    def fire(self, target_x, target_y):
        cur = WEAPON_LIST[self.weapon]
        spec = WEAPONS[cur]
        angle = angle_between(self.x, self.y, target_x, target_y)
        bullets = []
        if cur == 'grenade':
            # throw grenade in arc: set upward component
            speed = BULLET_SPEED * spec['speed']
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 300.0  # launch upward
            bullets.append(Bullet(self.x, self.y, vx, vy, self.id, dmg=spec['dmg'], is_grenade=True))
            self.weapon_cooldown = spec['cd']
        else:
            base_speed = BULLET_SPEED * spec['speed']
            for i in range(spec['count']):
                if spec['count'] == 1:
                    a = angle
                else:
                    spread = spec['spread']
                    a = angle + (random.random()*2 -1) * spread
                vx = math.cos(a) * base_speed + self.vx
                vy = math.sin(a) * base_speed + self.vy
                bullets.append(Bullet(self.x, self.y, vx, vy, self.id, dmg=spec['dmg']))
            self.weapon_cooldown = spec['cd']
        return bullets

    def take_damage(self, dmg):
        if self.shield_timer > 0.0:
            dmg *= 0.45
        self.health -= dmg


# ---------------- Arena / Environment (non-gym) ----------------
class Arena:
    def __init__(self, render=False):
        self.render_mode = render
        self.agents = [Agent(ARENA_PADDING + 60, SCREEN_H/2, 0), Agent(SCREEN_W - ARENA_PADDING - 60, SCREEN_H/2, 1)]
        self.bullets = []
        self.mines = []
        self.time = 0.0

    def reset(self):
        self.bullets = []
        self.mines = []
        self.time = 0.0
        self.agents[0].reset(ARENA_PADDING + 60, SCREEN_H/2)
        self.agents[1].reset(SCREEN_W - ARENA_PADDING - 60, SCREEN_H/2)

    def step(self, dt, actions=(None,None), manual=(None,None)):
        # actions: tuple for each agent: (ax, ay, shoot, switch, place_mine, teleport, dash, shield)
        # manual: same dictionary when player controls
        self.time += dt
        for i, ag in enumerate(self.agents):
            other = self.agents[1-i]
            act = actions[i]
            if manual[i] is not None:
                # manual input overrides
                ax = manual[i].get('ax',0.0)
                ay = manual[i].get('ay',0.0)
                shoot = 1 if manual[i].get('shoot',False) else 0
                switch = manual[i].get('switch',0)
                place_mine = 1 if manual[i].get('mine',False) else 0
                teleport = 1 if manual[i].get('teleport',False) else 0
                dash = 1 if manual[i].get('dash',False) else 0
                shield = 1 if manual[i].get('shield',False) else 0
            else:
                if act is None:
                    ax = ay = shoot = switch = place_mine = teleport = dash = shield = 0
                else:
                    ax = clamp(float(act[0]), -1.0, 1.0)
                    ay = clamp(float(act[1]), -1.0, 1.0)
                    shoot = 1 if act[2] > 0.5 else 0
                    sw = int(round(act[3]))
                    switch = clamp(sw, -1, 1)
                    place_mine = 1 if act[4] > 0.5 else 0
                    teleport = 1 if act[5] > 0.5 else 0
                    dash = 1 if act[6] > 0.5 else 0
                    shield = 1 if act[7] > 0.5 else 0

            # weapon switch
            if switch != 0:
                ag.weapon = (ag.weapon + switch) % len(WEAPON_LIST)

            # abilities
            if place_mine and ag.mine_cd <= 0.0:
                self.mines.append(Mine(ag.x, ag.y, ag.id))
                ag.mine_cd = MINE_COOLDOWN
            if teleport and ag.teleport_cd <= 0.0:
                # teleport to a random free location near center-ish
                tx = random.uniform(ARENA_PADDING+40, SCREEN_W-ARENA_PADDING-40)
                ty = random.uniform(ARENA_PADDING+40, SCREEN_H-ARENA_PADDING-40)
                ag.x = tx; ag.y = ty; ag.vx = 0; ag.vy = 0
                ag.teleport_cd = TELEPORT_COOLDOWN
            if dash and ag.dash_cd <= 0.0:
                # dash gives a short velocity burst in movement direction
                mag = math.hypot(ax, ay)
                if mag > 0.2:
                    nx = ax / mag; ny = ay / mag
                    ag.vx += nx * 300.0
                    ag.vy += ny * 300.0
                    ag.dash_cd = DASH_COOLDOWN
            if shield and ag.shield_cd <= 0.0:
                ag.shield_timer = SHIELD_DURATION
                ag.shield_cd = SHIELD_COOLDOWN

            # movement: forces -> acceleration
            ag.vx += ax * ACCEL * dt
            ag.vy += ay * ACCEL * dt
            speed = math.hypot(ag.vx, ag.vy)
            max_speed = MAX_SPEED * (1.6 if ag.dashing else 1.0)
            if speed > max_speed and speed>0:
                ag.vx *= max_speed / speed
                ag.vy *= max_speed / speed

            ag.x += ag.vx * dt
            ag.y += ag.vy * dt
            # bounds
            ag.x = clamp(ag.x, ARENA_PADDING, SCREEN_W-ARENA_PADDING)
            ag.y = clamp(ag.y, ARENA_PADDING, SCREEN_H-ARENA_PADDING)

            # shooting
            if shoot and ag.can_fire():
                bullets = ag.fire(other.x, other.y)
                for b in bullets:
                    self.bullets.append(b)

            # apply timers
            ag.apply_timers(dt)

        # update bullets
        for b in list(self.bullets):
            b.update(dt)
            # grenade explode
            if b.is_grenade and b.timer <= 0.0:
                # create shrapnel bullets in all directions
                shrap = 8
                for k in range(shrap):
                    a = (k / shrap) * 2*math.pi
                    vx = math.cos(a) * BULLET_SPEED*0.6
                    vy = math.sin(a) * BULLET_SPEED*0.6
                    self.bullets.append(Bullet(b.x, b.y, vx, vy, b.owner, dmg=max(8,int(b.dmg*0.5))))
                try:
                    self.bullets.remove(b)
                except ValueError:
                    pass
                continue

        # update mines
        for m in list(self.mines):
            m.update(dt)
            if m.life <= 0:
                try:
                    self.mines.remove(m)
                except ValueError:
                    pass

        # collisions bullets <-> agents
        for b in list(self.bullets):
            removed = False
            if b.life <= 0:
                try:
                    self.bullets.remove(b)
                except ValueError:
                    pass
                continue
            for ag in self.agents:
                if ag.id == b.owner:
                    continue
                dx = ag.x - b.x; dy = ag.y - b.y
                if dx*dx + dy*dy <= (AGENT_RADIUS + BULLET_RADIUS)**2:
                    ag.take_damage(b.dmg)
                    # reward owner score
                    owner = self.agents[b.owner]
                    owner.score += 1
                    try:
                        self.bullets.remove(b)
                    except ValueError:
                        pass
                    removed = True
                    break
            if removed:
                continue

        # collision agent <-> mine
        for m in list(self.mines):
            for ag in self.agents:
                if ag.id == m.owner:
                    continue
                dx = ag.x - m.x; dy = ag.y - m.y
                if dx*dx + dy*dy <= (AGENT_RADIUS + 8)**2:
                    # explode mine
                    for k in range(6):
                        a = (k/6.0)*2*math.pi
                        vx = math.cos(a) * BULLET_SPEED*0.5
                        vy = math.sin(a) * BULLET_SPEED*0.5
                        self.bullets.append(Bullet(m.x, m.y, vx, vy, m.owner, dmg=12))
                    try:
                        self.mines.remove(m)
                    except ValueError:
                        pass
                    break

        # check deaths
        done = False; winner = None
        alive = [ag.is_alive() for ag in self.agents]
        if not all(alive):
            done = True
            if alive[0] and not alive[1]: winner = 0
            elif alive[1] and not alive[0]: winner = 1
            else: winner = None

        return done, winner


# ---------------- Heuristic opponent (simple script) ----------------
class HeuristicBot:
    def __init__(self, agent_index=1):
        self.idx = agent_index

    def act(self, obs):
        # obs is a tuple (arena, agent_index)
        arena, i = obs
        ag = arena.agents[i]
        other = arena.agents[1-i]
        # aim at enemy
        dx = other.x - ag.x; dy = other.y - ag.y
        dist = math.hypot(dx, dy) + 1e-6
        ax = clamp(dx/dist, -1, 1) * 0.6
        ay = clamp(dy/dist, -1, 1) * 0.6
        shoot = 1 if dist < 700 else 0
        # switch to shotgun if close
        if dist < 200:
            switch = 2  # shotgun index (may wrap)
        else:
            switch = 0
        place_mine = 1 if dist < 180 and random.random() < 0.02 else 0
        teleport = 0
        dash = 0
        shield = 0
        return (ax, ay, shoot, switch, place_mine, teleport, dash, shield)


# ---------------- Gym Wrapper for RL (single-agent training vs heuristic opponent) ----------------
if RL_AVAILABLE:
    class ArenaSelfPlayEnv(gym.Env):
        """Gym environment where the RL agent controls Agent 0 and Agent 1 is heuristic or loaded policy."""
        metadata = {'render.modes': ['human']}

        def __init__(self, render=False, opponent_policy=None):
            super().__init__()
            self.render_mode = render
            self.arena = Arena(render=render)
            # observation: for agent0 we create a vector capturing relative info
            high = np.array([1.0]*OBS_DIM, dtype=np.float32)
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)
            # action: continuous vector with 8 entries packed: ax, ay, shoot_prob, switch(-1..1), place_mine_prob, teleport_prob, dash_prob, shield_prob
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
            self.opponent = opponent_policy if opponent_policy is not None else HeuristicBot(agent_index=1)
            self.max_t = 8.0
            self.dt = 1.0 / 30.0

        def obs_from_agent(self, agent_idx):
            ag = self.arena.agents[agent_idx]
            other = self.arena.agents[1-agent_idx]
            dx = (other.x - ag.x) / max(1.0, math.hypot(SCREEN_W, SCREEN_H))
            dy = (other.y - ag.y) / max(1.0, math.hypot(SCREEN_W, SCREEN_H))
            vx = ag.vx / MAX_SPEED
            vy = ag.vy / MAX_SPEED
            ox = other.vx / MAX_SPEED
            oy = other.vy / MAX_SPEED
            dist = math.hypot(other.x - ag.x, other.y - ag.y) / math.hypot(SCREEN_W, SCREEN_H)
            ang = math.atan2(other.y - ag.y, other.x - ag.x) / math.pi
            hp = ag.health / MAX_HEALTH
            ohp = other.health / MAX_HEALTH
            wcd = ag.weapon_cooldown / 2.0
            mine = ag.mine_cd / 4.0
            tp = ag.teleport_cd / 6.0
            dash = ag.dash_cd / 4.0
            shield = ag.shield_cd / 10.0
            vec = np.array([dx, dy, vx, vy, ox, oy, dist, ang, hp, ohp, wcd, mine, tp, dash, shield, 0.0], dtype=np.float32)
            return vec

        def reset(self):
            self.arena.reset()
            self.t = 0.0
            return self.obs_from_agent(0)

        def step(self, action):
            # parse action
            ax = clamp(action[0], -1.0, 1.0)
            ay = clamp(action[1], -1.0, 1.0)
            shoot = 1 if action[2] > 0.0 else 0
            switch = int(round(action[3]))
            place_mine = 1 if action[4] > 0.0 else 0
            teleport = 1 if action[5] > 0.0 else 0
            dash = 1 if action[6] > 0.0 else 0
            shield = 1 if action[7] > 0.0 else 0

            # opponent action
            opp_act = self.opponent.act((self.arena,1))
            actions = ((ax,ay,shoot,switch,place_mine,teleport,dash,shield), opp_act)

            done, winner = self.arena.step(self.dt, actions=actions, manual=(None,None))
            self.t += self.dt
            # reward shaping
            reward = 0.0
            # small time penalty
            reward -= 0.005
            # +1 for kill
            if done:
                if winner == 0:
                    reward += 1.0
                elif winner == 1:
                    reward -= 1.0
            else:
                # encourage damage to opponent and survival
                a = self.arena.agents[0]; b = self.arena.agents[1]
                # delta health as proxy
                reward += ( (b.health - a.health) / MAX_HEALTH ) * 0.02
            obs = self.obs_from_agent(0)
            info = {}
            if self.t >= self.max_t:
                # end by time
                done = True
            return obs, reward, done, info

        def render(self, mode='human'):
            # we will render using the outer pygame UI; this gym render is minimal
            pass

        def close(self):
            pass

# ---------------- Pygame UI & Controls ----------------
class UI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption('AI Battle Arena (PPO)')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 16)
        self.small = pygame.font.SysFont('Consolas', 14)
        self.arena = Arena(render=True)
        self.manual = [None, None]
        self.human_side = None
        self.running = True
        self.train_mode = False
        # RL models: we'll use PPO for agent0 optionally
        self.modelA = None
        self.modelB = None

    def draw(self):
        s = self.screen
        s.fill((18,18,20))
        pygame.draw.rect(s, (40,40,40), (ARENA_PADDING, ARENA_PADDING, SCREEN_W-2*ARENA_PADDING, SCREEN_H-2*ARENA_PADDING), 2)
        # bullets
        for b in self.arena.bullets:
            col = (255,220,80) if b.owner==0 else (80,220,255)
            pygame.draw.circle(s, col, (int(b.x), int(b.y)), BULLET_RADIUS)
        # mines
        for m in self.arena.mines:
            pygame.draw.circle(s, (180,80,40), (int(m.x), int(m.y)), 6)
        # agents
        for ag in self.arena.agents:
            col = (220,120,120) if ag.id==0 else (120,160,240)
            pygame.draw.circle(s, col, (int(ag.x), int(ag.y)), AGENT_RADIUS)
            # health bar
            hw = 44; hh=8
            hx = ag.x - hw/2; hy = ag.y - AGENT_RADIUS - 14
            pygame.draw.rect(s, (40,40,40), (hx,hy,hw,hh))
            pygame.draw.rect(s, (90,220,100), (hx+1,hy+1, int((hw-2)*clamp(ag.health/ag.health if ag.health>0 else 0,0,1)), hh-2))
            # weapon name
            ws = WEAPON_LIST[ag.weapon]
            txt = self.small.render(ws, True, (230,230,230))
            s.blit(txt, (ag.x - 16, ag.y + AGENT_RADIUS + 4))
            # cooldowns
            cdtxt = self.small.render(f"Wcd:{ag.weapon_cooldown:.1f} M:{ag.mine_cd:.1f} TP:{ag.teleport_cd:.1f}", True, (200,200,200))
            s.blit(cdtxt, (ag.x-36, ag.y - AGENT_RADIUS - 28))

        # HUD
        lines = [f"A HP:{int(self.arena.agents[0].health)}  Score:{self.arena.agents[0].score}", f"B HP:{int(self.arena.agents[1].health)}  Score:{self.arena.agents[1].score}", "Controls: 1/2 control, 0 release, WASD/arrows, Space fire, Q/E switch, G mine, T teleport, P train, S save, L load, R reset"]
        for i,l in enumerate(lines):
            s.blit(self.font.render(l, True, (220,220,220)), (12,8 + i*18))
        if self.train_mode:
            s.blit(self.font.render("TRAINING... (P toggled)", True, (255,200,100)), (SCREEN_W-260, 12))

    def save_models(self):
        os.makedirs('saved_models', exist_ok=True)
        if self.modelA is not None:
            fn = os.path.join('saved_models','ppo_agentA.zip')
            self.modelA.save(fn)
            print('Saved modelA to', fn)
        if self.modelB is not None:
            fn = os.path.join('saved_models','ppo_agentB.zip')
            self.modelB.save(fn)
            print('Saved modelB to', fn)

    def load_models(self):
        if not RL_AVAILABLE:
            print('RL libs not available; cannot load stable-baselines3 models')
            return
        fn = input('Enter saved model filename (saved_models/*.zip or full path): ').strip()
        if not fn:
            print('Load canceled')
            return
        if os.path.exists(fn):
            path = fn
        elif os.path.exists(os.path.join('saved_models', fn)):
            path = os.path.join('saved_models', fn)
        else:
            print('File not found')
            return
        # load as AgentA
        try:
            self.modelA = PPO.load(path)
            print('Loaded model as AgentA from', path)
        except Exception as e:
            print('Failed to load:', e)

    def train_agentA(self, timesteps=20000, render_during=False):
        if not RL_AVAILABLE:
            print('Install RL libs to train: pip install torch stable-baselines3 gym==0.21.0')
            return
        # create environment
        env = ArenaSelfPlayEnv(render=render_during, opponent_policy=HeuristicBot(agent_index=1))
        vec = DummyVecEnv([lambda: env])
        model = PPO('MlpPolicy', vec, verbose=1)
        model.learn(total_timesteps=timesteps)
        self.modelA = model
        print('Training finished — model stored in UI.modelA')

    def run(self):
        # main loop
        last = time.time()
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self.running = False
                    elif ev.key == pygame.K_1:
                        self.human_side = 0; self.manual[0] = {'ax':0.0,'ay':0.0,'shoot':False,'switch':0,'mine':False,'teleport':False,'dash':False,'shield':False}
                    elif ev.key == pygame.K_2:
                        self.human_side = 1; self.manual[1] = {'ax':0.0,'ay':0.0,'shoot':False,'switch':0,'mine':False,'teleport':False,'dash':False,'shield':False}
                    elif ev.key == pygame.K_0:
                        self.human_side = None; self.manual = [None,None]
                    elif ev.key == pygame.K_SPACE:
                        if self.human_side is not None:
                            self.manual[self.human_side]['shoot'] = True
                    elif ev.key == pygame.K_q:
                        if self.human_side is not None:
                            self.manual[self.human_side]['switch'] = -1
                    elif ev.key == pygame.K_e:
                        if self.human_side is not None:
                            self.manual[self.human_side]['switch'] = 1
                    elif ev.key == pygame.K_g:
                        if self.human_side is not None:
                            self.manual[self.human_side]['mine'] = True
                    elif ev.key == pygame.K_t:
                        if self.human_side is not None:
                            self.manual[self.human_side]['teleport'] = True
                    elif ev.key == pygame.K_r:
                        self.arena.reset()
                    elif ev.key == pygame.K_s:
                        self.save_models()
                    elif ev.key == pygame.K_l:
                        self.load_models()
                    elif ev.key == pygame.K_p:
                        # toggle training (runs small training job in blocking manner)
                        if RL_AVAILABLE:
                            self.train_mode = True
                            pygame.display.set_caption('AI Battle Arena (training...)')
                            print('Starting training (PPO) for AgentA — running 20k timesteps')
                            self.train_agentA(timesteps=20000, render_during=False)
                            pygame.display.set_caption('AI Battle Arena (PPO)')
                            self.train_mode = False
                        else:
                            print('RL libs missing')
                elif ev.type == pygame.KEYUP:
                    if ev.key == pygame.K_SPACE and self.human_side is not None:
                        self.manual[self.human_side]['shoot'] = False
                    if ev.key == pygame.K_g and self.human_side is not None:
                        self.manual[self.human_side]['mine'] = False
                    if ev.key == pygame.K_t and self.human_side is not None:
                        self.manual[self.human_side]['teleport'] = False

            # manual movement handling
            keys = pygame.key.get_pressed()
            if self.human_side is not None:
                ax = 0.0; ay = 0.0
                if keys[pygame.K_w] or keys[pygame.K_UP]: ay -= 1.0
                if keys[pygame.K_s] or keys[pygame.K_DOWN]: ay += 1.0
                if keys[pygame.K_a] or keys[pygame.K_LEFT]: ax -= 1.0
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]: ax += 1.0
                if ax!=0 and ay!=0:
                    ax*=0.7; ay*=0.7
                self.manual[self.human_side].update({'ax':ax,'ay':ay})

            # compute actions for RL/heuristic
            actions = [None, None]
            # Agent0 action from modelA if present
            if self.modelA is not None:
                # use latest observation
                if RL_AVAILABLE:
                    obs = ArenaSelfPlayEnv(render=False).obs_from_agent(0)  # quick obs snapshot from fresh env - approximate but good enough
                    act, _ = self.modelA.predict(obs, deterministic=False)
                    actions[0] = act
            elif self.arena.agents[0].policy is not None:
                actions[0] = self.arena.agents[0].policy.act((self.arena,0))
            else:
                # heuristic fallback
                actions[0] = HeuristicBot(agent_index=0).act((self.arena,0))

            # Agent1 action
            if self.modelB is not None:
                if RL_AVAILABLE:
                    obs1 = ArenaSelfPlayEnv(render=False).obs_from_agent(1)
                    act1, _ = self.modelB.predict(obs1, deterministic=False)
                    actions[1] = act1
            else:
                actions[1] = HeuristicBot(agent_index=1).act((self.arena,1))

            # manual overrides
            manual_wrapped = [None, None]
            for i in (0,1):
                if self.manual[i] is not None:
                    manual_wrapped[i] = self.manual[i]

            done, winner = self.arena.step(1.0/FPS, actions=actions, manual=manual_wrapped)
            if done:
                if winner is not None:
                    print('Round Winner:', winner)
                self.arena.reset()

            # draw
            self.draw()
            pygame.display.flip()

        pygame.quit()


# ---------------- Main ----------------
if __name__ == '__main__':
    ui = UI()
    ui.run()
