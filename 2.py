# question 2.
from commons import AbstractRLTask
from commons import AbstractAgent
import numpy as np
import matplotlib.pyplot as plt
import minihack_env as me
import random
import copy

# 2.
# state representation helper
def state_key(state):
    chars = state.get("chars") # using the "chars" representation
    return tuple(chars.flatten().tolist()) # flatten to tuple

class MCOnPolicyAgent(AbstractAgent):
    """ MC on-policy first-visit """
    def __init__(self, id, action_space, alpha=0, gamma=1, eps_start=0.1, eps_end=0.1, max_num_episodes=200):
        super().__init__(id, action_space)
        self.gamma = gamma  # discount
        self.alpha = alpha # dummy, not used

        self.eps_start, self.eps_end = eps_start, eps_end # in case eps_start = eps_end then epsilon is just eps_start
        self.episode_idx = 0
        self.max_num_episodes = max_num_episodes
        self.epsilon = eps_start  # initial epsilon to be overwritten

        self.Q = {}  # (s,a)
        self.Returns = {} # Returns for every (s,a) state action pair
        self.episode = [] # [(state0, action0, reward0), …]

    def act(self, state, reward=0):
        s = state_key(state)
        if s not in self.Q:
            self.Q[s] = np.zeros(self.action_space.n)
            for a in range(self.action_space.n):
                self.Returns[(s, a)] = []
        # epsilon-greedy
        if np.random.rand() < self.epsilon:
            a = self.action_space.sample() # random
        else:
            a = int(np.argmax(self.Q[s])) # greedy
        self.episode.append((s, a, reward))
        return a

    def onEpisodeEnd(self):
        G = 0
        visited = set()
        for s, a, r in reversed(self.episode):
            G = self.gamma * G + r
            if (s, a) not in visited: # only consider first-visit
                visited.add((s, a))
                self.Returns[(s, a)].append(G)
                self.Q[s][a] = np.mean(self.Returns[(s, a)])
        self.episode.clear()

        # Update epsilon
        self.episode_idx += 1
        # there will be decay in case eps_start!=eps_end, otherwise no decay
        self.epsilon = max(self.eps_end, self.eps_start - self.episode_idx * ((self.eps_start - self.eps_end)/self.max_num_episodes))

class SarsaAgent(AbstractAgent):
    """ SARSA """

    def __init__(self, id, action_space, alpha=0, gamma=1, eps_start=0.1, eps_end=0.1, max_num_episodes=200):
        super().__init__(id, action_space)
        self.gamma = gamma
        self.alpha = alpha  # step size, learning rate

        self.eps_start, self.eps_end = eps_start, eps_end
        self.episode_idx = 0
        self.max_num_episodes = max_num_episodes
        self.epsilon = eps_start

        self.Q = {}
        self.prev_state, self.prev_action = None, None

    def act(self, state, reward=0):
        s2 = state_key(state)
        # initialize Q for new state
        if s2 not in self.Q:
            self.Q[s2] = np.zeros(self.action_space.n)

        # Choose action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            a2 = self.action_space.sample()
        else:
            a2 = int(np.argmax(self.Q[s2]))

        if self.prev_state is not None: # if not the first, then update
            s1, a1 = self.prev_state, self.prev_action
            # epsilon greedy action is chosen for an update
            self.Q[s1][a1] += self.alpha * (reward + self.gamma * self.Q[s2][a2] - self.Q[s1][a1])
            # epsilon greedy is used both for an initialization of each episode and update over each step

        self.prev_state, self.prev_action = s2, a2
        return a2
    def onEpisodeEnd(self):
        self.prev_state, self.prev_action = None, None

        self.episode_idx += 1
        self.epsilon = max(self.eps_end, self.eps_start - self.episode_idx * ((self.eps_start - self.eps_end)
                                                                              / self.max_num_episodes))

    def epsilon_greedy(self, s):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # random
        return np.argmax(self.Q[s])  # else, greedy

class QLearningAgent(AbstractAgent):
    """ Q-Learning """
    def __init__(self, id, action_space, alpha=0, gamma=1, eps_start=0.1, eps_end=0.1, max_num_episodes=200):
        super().__init__(id, action_space)
        self.gamma = gamma
        self.alpha = alpha

        self.eps_start, self.eps_end = eps_start, eps_end
        self.episode_idx = 0
        self.max_num_episodes = max_num_episodes
        self.epsilon = eps_start

        self.Q = {}
        self.prev_state, self.prev_action = None, None

    def act(self, state, reward=0):
        s2 = state_key(state)
        if s2 not in self.Q:
            self.Q[s2] = np.zeros(self.action_space.n)

        # Choose action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            a2 = self.action_space.sample()
        else:
            a2 = int(np.argmax(self.Q[s2]))

        if self.prev_state is not None: # if it is not the first, then update
            s, a = self.prev_state, self.prev_action
            self.Q[s][a] += self.alpha * (reward + self.gamma * np.max(self.Q[s2]) - self.Q[s][a]) # greedy update

        self.prev_state, self.prev_action = s2, a2
        return a2

    def onEpisodeEnd(self):
        self.prev_state, self.prev_action = None, None

        self.episode_idx += 1
        self.epsilon = max(self.eps_end, self.eps_start - self.episode_idx * ((self.eps_start - self.eps_end)
                                                                              / self.max_num_episodes))

class RLTask(AbstractRLTask):
    def interact(self, n_episodes, track_episodes=None):
        returns, avg_returns = [], []
        cumulative_return = 0
        snapshots = {}   # ep_idx -> list of (row,col) pairs

        for ep in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                # NOTE: make sure `state` is still your dict here
                action = self.agent.act(state, total_reward)

                state, reward, term, trunc, info = self.env.step(action)
                total_reward += reward
                done = term or trunc

            self.agent.onEpisodeEnd()

            if track_episodes and ep in track_episodes:
                snapshots[ep] = copy.deepcopy(self.agent)
            returns.append(total_reward)
            cumulative_return += total_reward
            avg_returns.append(cumulative_return / (ep + 1))

        return returns, avg_returns, snapshots


def run(agent_cls):
    env  = me.get_minihack_envirnment(env_id)

    ag   = agent_cls(id="", action_space=env.action_space, alpha=alpha, eps_start=eps_start, eps_end=eps_end,
                     max_num_episodes=max_num_episodes)

    task = RLTask(env, ag)
    curve_returns, curve_avg_returns, _ = task.interact(n_episodes=episodes)
    return curve_returns, curve_avg_returns

def plot(agents):
    data = {}
    for agent_cls, label, colour in agents:
        curve_ret, curve_avg = run(agent_cls)
        data[label] = (colour, curve_ret, curve_avg)

    # 1) Average‐return curves
    plt.figure(figsize=(6, 4))
    for label, (colour, _, curve_avg) in data.items():
        plt.plot(curve_avg, label=label, color=colour)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.suptitle(f"Average return over {episodes} episodes on {env_id}")
    if eps_start == eps_end:
        plt.title(f"alpha={alpha}, epsilon={eps_start}, gamma={gamma}")
    else:
        plt.title(
            f"alpha={alpha}, eps_start={eps_start}, eps_end={eps_end}, "
            f"max_num_eps={max_num_episodes}, gamma={gamma}", fontsize=8
        )
    plt.tight_layout()
    plt.show()

    # 2) Single‐episode return curves
    plt.figure(figsize=(6, 4))
    for label, (colour, curve_ret, _) in data.items():
        plt.plot(curve_ret, label=label, color=colour)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.suptitle(f"Single return of every episode on {env_id}")
    if eps_start == eps_end:
        plt.title(f"alpha={alpha}, epsilon={eps_start}, gamma={gamma}")
    else:
        plt.title(
            f"alpha={alpha}, eps_start={eps_start}, eps_end={eps_end}, "
            f"max_num_eps={max_num_episodes}, gamma={gamma}", fontsize=8
        )
    plt.tight_layout()
    plt.show()

#2.1
episodes = 1000 # number of episodes
gamma = 1
alpha = 0.2
eps_start = 0.1
eps_end = 0.1
max_num_episodes=1000 # the maximum number of episodes used in the decay
# env
env_id = me.EMPTY_ROOM # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (MCOnPolicyAgent, "MC on-policy", "blue"),
    (SarsaAgent,      "SARSA",        "green"),
    (QLearningAgent,  "Q-learning",   "red")
]
plot(agents_to_compare)

episodes = 1000 # number of episodes
gamma = 1
alpha = 0.2
eps_start = 0.2
eps_end = 0.2
max_num_episodes=1000 # the maximum number of episodes used in the decay
# env
env_id = me.EMPTY_ROOM # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (MCOnPolicyAgent, "MC on-policy", "blue"),
    (SarsaAgent,      "SARSA",        "green"),
    (QLearningAgent,  "Q-learning",   "red")
]
plot(agents_to_compare)

episodes = 1000 # number of episodes
gamma = 1
alpha = 0.4
eps_start = 0.2
eps_end = 0.2
max_num_episodes=1000 # the maximum number of episodes used in the decay
# env
env_id = me.EMPTY_ROOM # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (MCOnPolicyAgent, "MC on-policy", "blue"),
    (SarsaAgent,      "SARSA",        "green"),
    (QLearningAgent,  "Q-learning",   "red")
]
plot(agents_to_compare)

episodes = 1000 # number of episodes
gamma = 1
alpha = 0.2
eps_start = 0.1
eps_end = 0.1
max_num_episodes=500 # the maximum number of episodes used in the decay
# env
env_id = me.ROOM_WITH_LAVA # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (MCOnPolicyAgent, "MC on-policy", "blue"),
    (SarsaAgent,      "SARSA",        "green"),
    (QLearningAgent,  "Q-learning",   "red")
]
plot(agents_to_compare)

episodes = 1000 # number of episodes
gamma = 0.9
alpha = 0.2
eps_start = 0.1
eps_end = 0.1
max_num_episodes=500 # the maximum number of episodes used in the decay
# env
env_id = me.ROOM_WITH_LAVA # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (MCOnPolicyAgent, "MC on-policy", "blue"),
    (SarsaAgent,      "SARSA",        "green"),
    (QLearningAgent,  "Q-learning",   "red")
]
plot(agents_to_compare)

episodes = 1000 # number of episodes
gamma = 1
alpha = 0.2
eps_start = 0.1
eps_end = 0.1
max_num_episodes=500 # the maximum number of episodes used in the decay
# env
env_id = me.ROOM_WITH_MONSTER # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (MCOnPolicyAgent, "MC on-policy", "blue"),
    (SarsaAgent,      "SARSA",        "green"),
    (QLearningAgent,  "Q-learning",   "red")
]
plot(agents_to_compare)

episodes = 1000 # number of episodes
gamma = 0.9
alpha = 0.2
eps_start = 0.1
eps_end = 0.1
max_num_episodes=500 # the maximum number of episodes used in the decay
# env
env_id = me.ROOM_WITH_MONSTER # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (MCOnPolicyAgent, "MC on-policy", "blue"),
    (SarsaAgent,      "SARSA",        "green"),
    (QLearningAgent,  "Q-learning",   "red")
]
plot(agents_to_compare)

episodes = 2000 # number of episodes
gamma = 0.9
alpha = 0.2
eps_start = 0.2
eps_end = 0.2
max_num_episodes=500 # the maximum number of episodes used in the decay
# env
env_id = me.CLIFF # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (MCOnPolicyAgent, "MC on-policy", "blue"),
    (SarsaAgent,      "SARSA",        "green"),
    (QLearningAgent,  "Q-learning",   "red")
]
plot(agents_to_compare)

#2.2
episodes = 2000 # number of episodes
gamma = 0.9
alpha = 0.2
eps_start = 0.3
eps_end = 0.1
max_num_episodes=1000 # the maximum number of episodes used in the decay
# env
env_id = me.CLIFF # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (MCOnPolicyAgent, "MC on-policy", "blue"),
    (SarsaAgent,      "SARSA",        "green"),
    (QLearningAgent,  "Q-learning",   "red")
]
plot(agents_to_compare)

def target_action(agent, state):
    s = state_key(state)
    if s not in agent.Q:
        agent.Q[s] = np.zeros(agent.action_space.n)
    if isinstance(agent, QLearningAgent):
        return int(np.argmax(agent.Q[s])) # greedy
    if np.random.rand() < agent.epsilon:
        return agent.action_space.sample() # epsilon-greedy
    return int(np.argmax(agent.Q[s]))

def sample_trajectory(agent, env, agent_char=ord('@')):
    state, _ = env.reset()
    traj = []
    done = False
    while not done:
        chars = state.get('chars')
        pos = np.argwhere(chars == agent_char)
        traj.append(tuple(pos[0]))
        a = target_action(agent, state)
        state, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
    return traj

def plot_snapshots(snapshots, env_id, model=""):
    plt.figure(figsize=(12, 12))
    for i, ep in enumerate(sorted(snapshots.keys()), 1):
        agent_snap = snapshots[ep]
        env = me.get_minihack_envirnment(env_id)
        traj = sample_trajectory(agent_snap, env)

        xs, ys = zip(*traj)
        plt.subplot(2, 2, i)
        plt.plot(ys, xs, marker='o', linestyle='-')
        plt.gca().invert_yaxis()
        plt.title(f"Episode {ep}")
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(f"Sample trajectories under target policy at checkpoints using {model}")
    plt.tight_layout()
    plt.show()

episodes = 3000
env_id = me.CLIFF
alpha = 0.1
gamma = 0.99
eps_start = 0.3
eps_end = 0.05
max_num_episodes = 100
checkpoints = [10, 100, 500, 2500]

# SARSA
ag = SarsaAgent(id="", action_space=me.get_minihack_envirnment(env_id).action_space, alpha=alpha, eps_start=eps_start, eps_end=eps_end,
               max_num_episodes=max_num_episodes)

task = RLTask(me.get_minihack_envirnment(env_id), ag)
_, _, snapshots_sarsa = task.interact(n_episodes=episodes, track_episodes=checkpoints)
plot_snapshots(snapshots_sarsa, env_id, model="SARSA")

# Q-learning
ag = QLearningAgent(id="", action_space=me.get_minihack_envirnment(env_id).action_space, alpha=alpha, eps_start=eps_start, eps_end=eps_end,
               max_num_episodes=max_num_episodes)

task = RLTask(me.get_minihack_envirnment(env_id), ag)
_, _, snapshots_q = task.interact(n_episodes=episodes, track_episodes=checkpoints)
plot_snapshots(snapshots_q, env_id, model="Q-learning")

#2.3
class DynaQAgent(QLearningAgent):
    """ Dyna-Q """
    def __init__(self, id, action_space, alpha=0, gamma=1, eps_start=0.1, eps_end=0.1, max_num_episodes=200,
                 planning_steps = 10):
        super().__init__(id, action_space, alpha=alpha, gamma=gamma, eps_start=eps_start, eps_end=eps_end,
                         max_num_episodes=max_num_episodes)

        self.model = {}  # model(s, a) -> (r, s')
        self.planning_steps = planning_steps

    def act(self, state, reward=0):
        s, a = self.prev_state, self.prev_action
        a2 = super().act(state, reward) # standard Q-learning update and select action
        s2 = state_key(state)
        if s is not None:
            self.model[(s, a)] = (reward, s2)
        # Planning updates
        for _ in range(self.planning_steps):
            if not self.model:
                break
            (s_p, a_p), (r_p, s_p_next) = random.choice(list(self.model.items()))
            self.Q[s_p][a_p] += self.alpha * (r_p + self.gamma * np.max(self.Q[s_p_next]) - self.Q[s_p][a_p])
        return a2

    def onEpisodeEnd(self):
        super().onEpisodeEnd()

episodes = 500 # number of episodes
gamma = 0.9
alpha = 0.2
eps_start = 0.1
eps_end = 0.1
max_num_episodes=500 # the maximum number of episodes used in the decay
# env
env_id = me.EMPTY_ROOM # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
# plot
agents_to_compare = [
    (QLearningAgent,  "Q-learning",   "red"),
    (DynaQAgent,      "Dyna-Q",       "purple"),
]
plot(agents_to_compare)

env_id = me.ROOM_WITH_LAVA # EMPTY_ROOM, CLIFF, ROOM_WITH_LAVA, ROOM_WITH_MONSTER
agents_to_compare = [
    (QLearningAgent,  "Q-learning",   "red"),
    (DynaQAgent,      "Dyna-Q",       "purple"),
]
plot(agents_to_compare)