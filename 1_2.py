# question 1.2
from commons import AbstractAgent
import numpy as np
import minihack_env as me
import commons
import matplotlib.pyplot as plt

class FixedAgent(AbstractAgent):
    def __init__(self, id, action_space):
        super().__init__(id, action_space)
        self.down = 2
        self.right = 1
        self.last_action = self.down

    def act(self, state, reward=0):
        chars = state["chars"]
        player_pos = np.argwhere(chars == ord('@'))  #the player
        if player_pos.size == 0: #no player
            return self.down
        x, y = player_pos[0]
        walkable = [ord('.'), ord('>')] #floor or goal
        #If possible, go down
        if x + 1 < chars.shape[0] and chars[x + 1, y] in walkable:
            self.last_action = self.down
            return self.down
        #Otherwise, go right if possible
        if y + 1 < chars.shape[1] and chars[x, y + 1] in walkable:
            self.last_action = self.right
            return self.right
        #Else, try right
        return self.right

def run_fixed_agent(env_id, title, max_ep=10):
    env = me.get_minihack_envirnment(env_id, add_pixel=True)
    agent = FixedAgent("fixed_agent", env.action_space)
    obs, _ = env.reset()
    print(f"\n{title}")
    print_ascii_map(obs, step=0, title=f"{title}")

    for t in range(max_ep):
        action = agent.act(obs)
        obs, reward, terminated, _, _ = env.step(action)
        print(f"\nStep {t+1}: Action = {action}, Reward = {reward}")
        if terminated:
            print("Episode ended.")
            break
        print_ascii_map(obs, step=t+1, title=f"{title}")

def print_ascii_map(obs, step=None, title=""):
    chars = commons.get_crop_chars_from_observation(obs)
    pixels = commons.get_crop_pixel_from_observation(obs)
    plt.figure()
    plt.imshow(pixels)
    plt.axis("off")
    if step is not None:
        plt.title(f"{title} – step {step}")
    plt.show()
    print('\n'.join(''.join(chr(c) for c in row) for row in chars))

run_fixed_agent(me.EMPTY_ROOM, "EMPTY_ROOM")
run_fixed_agent(me.ROOM_WITH_LAVA, "ROOM_WITH_LAVA")