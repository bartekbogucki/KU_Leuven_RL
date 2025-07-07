# question 3.
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import minihack_env as me
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
import os

for sub in ["dqn_empty", "ppo_empty", "dqn_empty2", "ppo_empty2", "dqn_monsters", "ppo_monsters"]:
   os.makedirs(f"./monitors/{sub}", exist_ok=True)

def make_env(env_id, log_dir=None):
    env = me.get_minihack_envirnment(env_id)
    env = gym.wrappers.FlattenObservation(env) # Flatten
    if log_dir is not None:
        env = Monitor(env, log_dir, allow_early_resets=True)
    return env

def train_agent(algorithm, env_id, policy="MlpPolicy", total_timesteps=1000000, log_dir=None, **kwargs):
    env = make_env(env_id, log_dir)
    model = algorithm(policy, env, verbose=1, **kwargs)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    duration = time.time() - t0
    return model, duration

# DQN
dqn_kwargs = dict(
    learning_rate=1e-4,
    buffer_size=25000,
    learning_starts=100,
    batch_size=32,
    train_freq=2,
    gradient_steps=2,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    policy_kwargs=dict(net_arch=[32, 32])
)

# PPO as actor-critic
ppo_kwargs = dict(
    learning_rate=1e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

dqn_empty, t_dqn_empty = train_agent(DQN, me.EMPTY_ROOM, total_timesteps=100000, policy="MlpPolicy",
                                    log_dir="./monitors/dqn_empty", **dqn_kwargs)

ppo_empty, t_ppo_empty = train_agent(PPO, me.EMPTY_ROOM, total_timesteps=100000, policy="MlpPolicy",
                                    log_dir="./monitors/ppo_empty", **ppo_kwargs)

# DQN
dqn_kwargs2 = dict(
    learning_rate=5e-5,
    buffer_size=25000,
    learning_starts=100,
    batch_size=32,
    train_freq=2,
    gradient_steps=2,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    policy_kwargs=dict(net_arch=[32, 32])
)

# PPO as actor-critic
ppo_kwargs2 = dict(
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=5,
    gamma=0.95,
    gae_lambda=0.90,
    ent_coef=0.01
)

dqn_empty2, t_dqn_empty2 = train_agent(DQN, me.EMPTY_ROOM, total_timesteps=150000, policy="MlpPolicy",
                                    log_dir="./monitors/dqn_empty2", **dqn_kwargs2)

ppo_empty2, t_ppo_empty2 = train_agent(PPO, me.EMPTY_ROOM, total_timesteps=150000, policy="MlpPolicy",
                                    log_dir="./monitors/ppo_empty2", **ppo_kwargs2)

dqn_monsters, t_dqn_mon = train_agent(DQN, me.ROOM_WITH_MULTIPLE_MONSTERS, total_timesteps=250000, policy="MlpPolicy",
                                     log_dir="./monitors/dqn_monsters", **dqn_kwargs2)

ppo_monsters, t_ppo_mon = train_agent(PPO, me.ROOM_WITH_MULTIPLE_MONSTERS, total_timesteps=250000, policy="MlpPolicy",
                                     log_dir="./monitors/ppo_monsters", **ppo_kwargs2)

def plot_learning(log_paths, algorithms, env, smooth_window=10, by_steps=False):
    plt.figure(figsize=(8,5))
    for path, algorithm in zip(log_paths, algorithms):
        df = pd.read_csv(path, comment='#')

        if by_steps:
            x = df['l'].cumsum()
            xlabel = 'Environment steps'
        else:
            x = np.arange(1, len(df)+1)
            xlabel = 'Episode'

        plt.plot(x, df['r'], alpha=0.3, label=f"{algorithm} (raw)")
        if smooth_window is not None: # smoothed reward, here instead of cumulative
            r_smooth = df['r'].rolling(window=smooth_window, min_periods=1).mean()
            plt.plot(x, r_smooth, label=f"{algorithm} (smoothed avg of {smooth_window} window)")
        std_last_100 = df["r"].tail(100).std()
        mean_last_100 = df["r"].tail(100).mean()
        print(f"Mean of {algorithm} on {env} based on last 100 episodes: {mean_last_100:.2f}")
        print(f"Standard deviation of {algorithm} on {env} based on last 100 episodes: {std_last_100:.2f}")

    plt.xlabel(xlabel)
    plt.ylabel('Reward')
    plt.title(f"Learning curves on {env}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

plot_learning(
    ["./monitors/dqn_empty/monitor.csv","./monitors/ppo_empty/monitor.csv"],
    ["DQN", "PPO"],
    "empty room",
    smooth_window=20,
    by_steps=False
)

plot_learning(
    ["./monitors/dqn_empty/monitor.csv","./monitors/ppo_empty/monitor.csv"],
    ["DQN", "PPO"],
    "empty room",
    smooth_window=20,
    by_steps=True
)

plot_learning(
    ["./monitors/dqn_empty2/monitor.csv","./monitors/ppo_empty2/monitor.csv"],
    ["DQN", "PPO"],
    "empty room",
    smooth_window=20,
    by_steps=False
)

plot_learning(
    ["./monitors/dqn_empty2/monitor.csv","./monitors/ppo_empty2/monitor.csv"],
    ["DQN", "PPO"],
    "empty room",
    smooth_window=20,
    by_steps=True
)

plot_learning(
    ["./monitors/dqn_monsters/monitor.csv","./monitors/ppo_monsters/monitor.csv"],
    ["DQN", "PPO"],
    "room with multiple monsters",
    smooth_window=20,
    by_steps=False
)

plot_learning(
    ["./monitors/dqn_monsters/monitor.csv","./monitors/ppo_monsters/monitor.csv"],
    ["DQN", "PPO"],
    "room with multiple monsters",
    smooth_window=20,
    by_steps=True
)

SAVE_DIR = "./saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

dqn_empty.save(f"{SAVE_DIR}/dqn_empty_final")
ppo_empty.save(f"{SAVE_DIR}/ppo_empty_final")

dqn_empty2.save(f"{SAVE_DIR}/dqn_empty2_final")
ppo_empty2.save(f"{SAVE_DIR}/ppo_empty2_final")

dqn_monsters.save(f"{SAVE_DIR}/dqn_monsters_final")
ppo_monsters.save(f"{SAVE_DIR}/ppo_monsters_final")

# If we want to load trained models, good to consider a loop with loading if already trained rather than above training
# Loading can be done as below
#loaded_dqn = DQN.load(f"{SAVE_DIR}/dqn_empty_final", env=env)