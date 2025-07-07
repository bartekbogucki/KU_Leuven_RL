# question 1.1
# 0. imports
from gymnasium import Env, spaces
import numpy as np
from commons import AbstractRLTask
from commons import AbstractAgent
import matplotlib.pyplot as plt

#1.
class GridWorldEnv(Env):
    def __init__(self, n=5, m=5):
        self.n = n
        self.m = m
        self.observation_space = spaces.Box(low=0, high=max(n, m), shape=(2,)) #two-dimensional cube
        self.action_space = spaces.Discrete(4)  # 4 moves
        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = np.array([0, 0])
        return self.agent_pos.copy(), {}

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0: #up and cannot perform whether if not valid
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.n - 1: #down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0: #left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.m - 1: #right
            self.agent_pos[1] += 1

        terminated = np.array_equal(self.agent_pos, [self.n - 1, self.m - 1]) #if goal then terminated
        truncated = False #no timelimit
        reward = -1.0 #always when step reward is -1
        return self.agent_pos.copy(), reward, terminated, truncated, {}

    def render(self):
        grid = [["." for _ in range(self.m)] for _ in range(self.n)]
        x, y = self.agent_pos
        grid[x][y] = "A" #overwrite . with A and G
        grid[self.n - 1][self.m - 1] = "G"
        print("\n".join([" ".join(row) for row in grid]))
        print()

#2.
class RandomAgent(AbstractAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()

#3.
class RLTask(AbstractRLTask):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def interact(self, episodes=10000):
        returns = []
        cumulative_return = 0
        for k in range(episodes):
            s, _ = self.env.reset()
            terminated = False
            total_reward = 0 #single episode reward
            while not terminated:
                action = self.agent.act(s)
                s, reward, terminated, _, _ = self.env.step(action)
                total_reward += reward
            cumulative_return += total_reward
            avg_return = cumulative_return / (k + 1)
            returns.append(avg_return)
        return returns

    def visualize_episode(self, max_steps=10):
        obs, _ = self.env.reset()
        self.env.render()
        for _ in range(max_steps):
            action = self.agent.act(obs)
            obs, reward, terminated, _, _ = self.env.step(action)
            self.env.render()
            if terminated:
                break

env = GridWorldEnv(n=5, m=5)
agent = RandomAgent(env.action_space)
task = RLTask(env, agent)

# Plot average returns
returns = task.interact(episodes=10000)
plt.plot(returns)
plt.title("Average return over an interaction of 10000 episodes")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.show()

print(f"Final average return after {len(returns)} episodes: {returns[-1]:.2f}")

# Visualize a sample episode
task.visualize_episode()