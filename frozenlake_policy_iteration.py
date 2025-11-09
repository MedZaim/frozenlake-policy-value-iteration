#%%
# Policy Iteration on FrozenLake-v1 (deterministic)
import numpy as np
import gymnasium as gym
import time

#%%
# Create env with human rendering so a window appears when stepping
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Access MDP transitions and spaces
P = env.unwrapped.P  # dict: P[s][a] -> list of (prob, next_state, reward, terminated)
nS = env.observation_space.n
nA = env.action_space.n

# Hyperparameters
gamma = 0.99
theta = 1e-8

#%%
# Policy evaluation: given a policy pi (array of action indices), compute V

def policy_evaluation(pi, V=None, gamma: float = gamma, theta: float = theta):
    if V is None:
        V = np.zeros(nS, dtype=np.float64)
    else:
        V = np.array(V, dtype=np.float64, copy=True)

    while True:
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            a = pi[s]
            v_new = 0.0
            for (prob, ns, r, done) in P[s][a]:
                v_new += prob * (r + gamma * (0.0 if done else V[ns]))
            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))
        if delta < theta:
            break
    return V

#%%
# Policy improvement: greedy w.r.t. V

def policy_improvement(V, gamma: float = gamma):
    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        q = np.zeros(nA, dtype=np.float64)
        for a in range(nA):
            for (prob, ns, r, done) in P[s][a]:
                q[a] += prob * (r + gamma * (0.0 if done else V[ns]))
        pi[s] = int(np.argmax(q))
    return pi

#%%
# Full policy iteration loop

def policy_iteration(gamma: float = gamma, theta: float = theta):
    # Step 1: Initialize policy randomly and value function
    pi = np.random.randint(0, nA, size=nS, dtype=int)
    V = np.zeros(nS, dtype=np.float64)

    iteration = 0
    while True:
        iteration += 1

        # Step 2: Evaluate current policy
        V = policy_evaluation(pi, V, gamma=gamma, theta=theta)

        # Step 3: Improve the policy using the current value function
        new_pi = policy_improvement(V, gamma=gamma)

        # Step 4: Check if policy has changed
        policy_stable = np.array_equal(pi, new_pi)

        # Step 5: Update policy
        pi = new_pi

        # Step 6: Stop if stable (converged)
        if policy_stable:
            break

    return pi, V, iteration



#%%
# Execute and display results

if __name__ == "main" or __name__ == "__main__":
    pi_opt, V_opt, iters = policy_iteration()
    print(f"Converged in {iters} iterations")
    side = int(np.sqrt(nS))
    print("Optimal V (reshaped if square):")
    if side * side == nS:
        print(np.round(V_opt.reshape(side, side), 3))
    else:
        print(np.round(V_opt, 3))

    arrow_map = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    print("\nOptimal Policy:")
    if side * side == nS:
        grid = np.array([arrow_map[a] for a in pi_opt]).reshape(side, side)
        for r in range(side):
            print(" ".join(grid[r]))
    else:
        print(pi_opt)

    # Run one episode to ensure a window appears
    def run_episode(env, pi):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            a = int(pi[obs])
            obs, r, terminated, truncated, info = env.step(a)
            total_reward += r
            steps += 1
        return total_reward, steps, terminated, truncated

    total_reward, steps, terminated, truncated = run_episode(env, pi_opt)
    print(f"Episode -> reward: {total_reward}, steps: {steps}, terminated: {terminated}, truncated: {truncated}")

    # Keep window visible briefly and pump events to avoid "Not Responding"
    try:
        import pygame
        for _ in range(90):  # ~3 seconds
            pygame.event.pump()
            time.sleep(0.03)
    except Exception:
        time.sleep(3.0)

    env.close()
