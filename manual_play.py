
import gym
import numpy as np
import time
import pygame

pygame.init()

# Build deterministic FrozenLake
custom_map = [
    "SFFHF",
    "HFFFF",
    "FFHFF",
    "HHFHF",
    "FFFGF"
]
env = gym.make("FrozenLake-v1", is_slippery=False, desc=custom_map, render_mode="human")

# Define sizes before value iteration
nS = env.observation_space.n
nA = env.action_space.n

def value_iteration(gamma=0.99, theta=1e-8):
    P = env.unwrapped.P
    V = np.zeros(nS, dtype=float)
    while True:
        delta = 0.0
        for s in range(nS):
            q_best = -1e9
            for a in range(nA):
                q = 0.0
                for (prob, ns, r, done) in P[s][a]:
                    q += prob * (r + gamma * (0.0 if done else V[ns]))
                if q > q_best:
                    q_best = q
            delta = max(delta, abs(q_best - V[s]))
            V[s] = q_best
        if delta < theta:
            break
    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        q = np.zeros(nA)
        for a in range(nA):
            for (prob, ns, r, done) in P[s][a]:
                q[a] += prob * (r + gamma * (0.0 if done else V[ns]))
        pi[s] = int(np.argmax(q))
    return pi, V

pi_opt, V_opt = value_iteration()

def _pump_events():
    try:
        pygame.event.pump()
    except Exception:
        pass

def _hold_window(seconds=3):
    t0 = time.time()
    while time.time() - t0 < seconds:
        _pump_events()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
        time.sleep(0.02)

def manual_then_auto(policy, delay=0.35):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    auto = False
    action_map = {
        pygame.K_LEFT: 0,
        pygame.K_DOWN: 1,
        pygame.K_RIGHT: 2,
        pygame.K_UP: 3
    }
    while not (terminated or truncated):
        if not auto:
            a = None
            while True:
                _pump_events()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
                        if event.key == pygame.K_RETURN:
                            auto = True
                            break
                        if event.key in action_map:
                            a = action_map[event.key]
                            break
                if auto or a is not None:
                    break
                time.sleep(0.02)
            if auto:
                continue
            obs, r, terminated, truncated, _ = env.step(a)
        else:
            a = int(policy[obs])
            obs, r, terminated, truncated, _ = env.step(a)
            _pump_events()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return
            time.sleep(delay)
    _hold_window()

if __name__ == "__main__":
    manual_then_auto(pi_opt)
    env.close()
