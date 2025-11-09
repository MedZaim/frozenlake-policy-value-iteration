# frozen_backend.py
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional

ACTION_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # LEFT, DOWN, RIGHT, UP

@dataclass
class EnvConfig:
    rows: int = 5
    cols: int = 5
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    holes: List[Tuple[int, int]] = None
    slippery: bool = False
    slip_prob: float = 0.2  # probability to ignore chosen action and pick random
    gamma: float = 0.99
    theta: float = 1e-8

    def __post_init__(self):
        if self.holes is None:
            self.holes = [(0, 3), (1, 0), (2, 2), (3, 0), (3, 2)]

class FrozenLakeCustom:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rows = cfg.rows
        self.cols = cfg.cols
        self.start = cfg.start
        self.goal = cfg.goal
        self.holes = set(cfg.holes)
        self.state = self._rc_to_index(*self.start)
        self.terminated = False

    def reset(self) -> int:
        self.state = self._rc_to_index(*self.start)
        self.terminated = False
        return self.state

    def index_to_rc(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.cols)

    def _rc_to_index(self, r: int, c: int) -> int:
        return r * self.cols + c

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_terminal_rc(self, r: int, c: int) -> bool:
        return (r, c) == self.goal or (r, c) in self.holes

    def step(self, action: int) -> Tuple[int, float, bool]:
        if self.terminated:
            return self.state, 0.0, True
        chosen = action
        if self.cfg.slippery:
            if np.random.rand() < self.cfg.slip_prob:
                chosen = np.random.randint(0, 4)
        r, c = self.index_to_rc(self.state)
        dr, dc = ACTION_DIRS[chosen]
        nr, nc = r + dr, c + dc
        if not self.in_bounds(nr, nc):
            nr, nc = r, c
        reward = 0.0
        if (nr, nc) == self.goal:
            reward = 1.0
        if (nr, nc) in self.holes:
            reward = 0.0
        self.state = self._rc_to_index(nr, nc)
        self.terminated = self.is_terminal_rc(nr, nc)
        return self.state, reward, self.terminated

    def build_transition_model(self):
        """Return P[s][a] = list of (prob, next_s, reward, done)."""
        nS = self.rows * self.cols
        P = [[[] for _ in range(4)] for _ in range(nS)]
        for s in range(nS):
            r, c = self.index_to_rc(s)
            terminal = self.is_terminal_rc(r, c)
            for a in range(4):
                if terminal:
                    P[s][a].append((1.0, s, 0.0, True))
                    continue
                outcomes = {}
                def add_out(act, base_prob):
                    rr, cc = r, c
                    dr, dc = ACTION_DIRS[act]
                    nr, nc = rr + dr, cc + dc
                    if not self.in_bounds(nr, nc):
                        nr, nc = rr, cc
                    ns = self._rc_to_index(nr, nc)
                    rew = 1.0 if (nr, nc) == self.goal else 0.0
                    done = self.is_terminal_rc(nr, nc)
                    key = (ns, rew, done)
                    outcomes[key] = outcomes.get(key, 0.0) + base_prob
                if self.cfg.slippery:
                    p_slip = self.cfg.slip_prob
                    p_main = 1 - p_slip
                    add_out(a, p_main)
                    slip_each = p_slip / 4.0
                    for alt in range(4):
                        add_out(alt, slip_each)
                else:
                    add_out(a, 1.0)
                for (ns, rew, done), prob in outcomes.items():
                    P[s][a].append((prob, ns, rew, done))
        return P

def value_iteration(env: FrozenLakeCustom, gamma: Optional[float] = None, theta: Optional[float] = None):
    gamma = gamma if gamma is not None else env.cfg.gamma
    theta = theta if theta is not None else env.cfg.theta
    P = env.build_transition_model()
    nS = env.rows * env.cols
    nA = 4
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
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        q = np.zeros(nA)
        for a in range(nA):
            for (prob, ns, r, done) in P[s][a]:
                q[a] += prob * (r + gamma * (0.0 if done else V[ns]))
        policy[s] = int(np.argmax(q))
    return policy, V