import random
import gym
import numpy as np
from gym_minigrid.wrappers import RGBImgObsWrapper, NESWActionsImage
from matplotlib import pyplot as plt

env = gym.make("MiniGrid-DoorKey-10x10-v1")
env = RGBImgObsWrapper(env)
env = NESWActionsImage(env, 0, 10000)

env.seed(0)

s = env.reset()

env.env.put_agent(1, 1, 0)

# for i in range(100):
#     action = random.choice(list(range(env.n_actions)))
#     s_, r, done, info = env.step(action)
#     print(env.key, env.door)
#     print(i, done, r)

image = np.rot90(env.get_grid_image())

print(s["pos"])
plt.imshow(image)
plt.show()

