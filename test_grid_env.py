import gym
from gym_minigrid.wrappers import RGBImgObsWrapper, NESWActionsImage
from matplotlib import pyplot as plt
import itertools
from utils import xy_2_image

env = gym.make("MiniGrid-NineRoomsDet-v0")
env = RGBImgObsWrapper(env)
env = NESWActionsImage(env, 0, 100)

room1_x = list(range(1, 6))
room1_y = list(range(1, 6))
room1 = list(itertools.product(room1_x, room1_y))

room2_x = list(range(7, 12))
room2_y = list(range(1, 6))
room2 = list(itertools.product(room2_x, room2_y))

room3_x = list(range(13, 18))
room3_y = list(range(1, 6))
room3 = list(itertools.product(room3_x, room3_y))

room4_x = list(range(1, 6))
room4_y = list(range(7, 12))
room4 = list(itertools.product(room4_x, room4_y))

room5_x = list(range(7, 12))
room5_y = list(range(7, 12))
room5 = list(itertools.product(room5_x, room5_y))

room6_x = list(range(13, 18))
room6_y = list(range(7, 12))
room6 = list(itertools.product(room6_x, room6_y))

room7_x = list(range(1, 6))
room7_y = list(range(13, 18))
room7 = list(itertools.product(room7_x, room7_y))

room8_x = list(range(7, 12))
room8_y = list(range(13, 18))
room8 = list(itertools.product(room8_x, room8_y))

room9_x = list(range(13, 18))
room9_y = list(range(13, 18))
room9 = list(itertools.product(room9_x, room9_y))

corridor_room1_4 = ((1, 4), [(5, 3), (6, 3), (7, 3)])
corridor_room1_2 = ((1, 2), [(3, 5), (3, 6), (3, 7)])
corridor_room4_7 = ((4, 7), [(11, 3), (12, 3), (13, 3)])
corridor_room4_5 = ((4, 5), [(9, 5), (9, 6), (9, 7)])
corridor_room2_5 = ((2, 5), [(5, 9), (6, 9), (7, 9)])
corridor_room2_3 = ((2, 3), [(3, 11), (3, 12), (3, 13)])
corridor_room5_8 = ((5, 8), [(11, 9), (12, 9), (13, 9)])
corridor_room5_6 = ((5, 6), [(9, 11), (9, 12), (9, 13)])
corridor_room3_6 = ((3, 6), [(5, 15), (6, 15), (7, 15)])
corridor_room6_9 = ((6, 9), [(11, 15), (12, 15), (13, 15)])


for xy in corridor_room6_9[1]:
    image = xy_2_image(env, xy)
    plt.imshow(image)
    plt.show()