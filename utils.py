import itertools
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import wandb
from gym_minigrid.wrappers import *
from replay import ExperienceReplay, ExperienceReplay_v2
import torch


def q_ev(state, action, next_state, reward, done):
    if state not in q_ev.Q_s_a_dict:
        q_ev.Q_s_a_dict[state] = {}
    if next_state not in q_ev.Q_s_a_dict:
        q_ev.Q_s_a_dict[next_state] = {}
    if action not in q_ev.Q_s_a_dict[state]:
        q_ev.Q_s_a_dict[state][action] = 0

    if done is True:
        td_error = reward
    else:
        td_error = reward + q_ev.gamma * \
                   (max(q_ev.Q_s_a_dict[next_state].values()) if len(q_ev.Q_s_a_dict[next_state].values()) > 0 else 0.)

    q_ev.Q_s_a_dict[state][action] += q_ev.alpha * (td_error - q_ev.Q_s_a_dict[state][action])

    return q_ev.Q_s_a_dict


q_ev.Q_s_a_dict = {}
q_ev.gamma = 0.99
q_ev.alpha = 1


class Option:
    # single action options for now

    init_set = set()
    terminal_set = set()
    action = None
    counter = 0
    id = None

    def __init__(self, init_set, action, terminal_set, as_s, as_t):
        self.init_set = init_set
        self.action = action
        self.terminal_set = terminal_set
        Option.counter += 1
        self.id = Option.counter
        self.as_s = as_s
        self.as_t = as_t

        self.state_tmp = None

    def get_action(self, state):
        # if state in self.terminal_set:     # WARNING WARNING WARNING WARNING WARNING WARNING WARNING
        #     return 4 # None action
        # else:
        return self.action

    def done(self, state):
        if state in self.terminal_set:
            return True
        else:
            return False

    def __str__(self):
        return "Option n:" + str(self.id)


class StocasticOption:
    # single action options for now

    init_set = set()
    terminal_set = set()
    action = None
    counter = 0
    id = None

    def __init__(self, init_set, action, terminal_set, as_s, as_t, stocastic_action_prob, n_actions):
        self.init_set = init_set
        self.action = action
        self.terminal_set = terminal_set
        Option.counter += 1
        self.id = Option.counter
        self.as_s = as_s
        self.as_t = as_t
        self.stocastic_action_prob = stocastic_action_prob
        self.state_tmp = None
        self.n_actions = n_actions

    def get_action(self, state):
        # if state in self.terminal_set:     # WARNING WARNING WARNING WARNING WARNING WARNING WARNING
        #     return 4 # None action
        # else:
        if random.random() > self.stocastic_action_prob:
            return self.action
        else:
            return random.choice(list(range(self.n_actions)))

    def done(self, state):
        if state in self.terminal_set:
            return True
        else:
            return False

    def __str__(self):
        return "Option n:" + str(self.id)


def print_grid_v(q, lim_min, lim_max, distance, cluster_round_value, keydoor=False):
    print("")
    if len(print_grid_v.all_states) == 0:
        for i in range(lim_min, lim_max):
            for j in range(lim_min, lim_max):
                print_grid_v.all_states.append((i, j))

    if keydoor is True:

        # no key no door
        for state in print_grid_v.all_states:
            x, y = state
            state = (x, y, 0, 0)
            if state in q:
                print((max(q[state].values()) if len(q[state].values()) > 0 else 0.), end=" ")
            else:
                print("###", end=" ")

            if state[1] == lim_max - 1:
                print("")
        print("")
        print("")
        print("")
        # key no door
        for state in print_grid_v.all_states:
            x, y = state
            state = (x, y, 1, 0)
            if state in q:
                print((max(q[state].values()) if len(q[state].values()) > 0 else 0.), end=" ")
            else:
                print("###", end=" ")

            if state[1] == lim_max - 1:
                print("")
        print("")
        print("")
        print("")
        # key door
        for state in print_grid_v.all_states:
            x, y = state
            state = (x, y, 1, 1)
            if state in q:
                print((max(q[state].values()) if len(q[state].values()) > 0 else 0.), end=" ")
            else:
                print("###", end=" ")

            if state[1] == lim_max - 1:
                print("")

    else:
        for state in print_grid_v.all_states:
            if state in q:
                print((max(q[state].values()) if len(q[state].values()) > 0 else 0.), end=" ")
            else:
                print("##", end=" ")

            if state[1] == lim_max - 1:
                print("")
        print("")
        print("")
        print("")


print_grid_v.all_states = []


def evaluate_q_policy(q, env, n_ep_ev=100, rendering=True):
    sum_reward = 0
    for i in range(n_ep_ev):
        s = env.reset()
        while True:
            # optimal policy
            action = max(q[s].items(), key=operator.itemgetter(1))[0]

            next_s, reward, done, info = env.step(action)
            if rendering:
                env.render()

            # end conditions
            s = next_s
            if done:
                sum_reward += reward
                break

    return sum_reward / n_ep_ev


def collect_transitions_and_reward(s, a, next_state, reward):
    collect_transitions_and_reward.t_r.add((s, a, next_state, reward))
    return collect_transitions_and_reward.t_r


def collect_transition_and_reward_environment(env, n_ep=100):
    t_r = set()
    for i in range(n_ep):
        s = env.reset()
        while True:
            # random policy
            action = int(random.random() * env.action_space.n)

            next_s, reward, done, info = env.step(action)
            t_r.add((s, action, next_s, reward))

            # end conditions
            s = next_s

            if done:
                break
    return t_r


def collect_option_trajectories(env, set_options, n_steps=1000, init_set=None, render=True):

    for _ in range(n_steps):
        trajectory = []
        action_trajectory = []
        if init_set is None:
            s = env.reset()
            trajectory.append(s)
            if render:
                env.render()
        else:
            init_state = random.choice(init_set)
            env.reset()
            env.env.put_agent(init_state[0], init_state[1], 0)
            s = init_state
            trajectory.append(s)
            if render:
                env.render()

        while True:
            option = random.choice(set_options)
            print(option)
            while True:
                action = option.get_action(s)
                next_s, reward, done, info = env.step(action)
                print(s, next_s, reward, option.done(next_s))
                if render:
                    env.render()
                    import time
                    time.sleep(1)

                trajectory.append(next_s)
                action_trajectory.append(action)

                # end conditions
                s = next_s
                if done or option.done(next_s):
                    break

            collect_option_trajectories.o_t.append(((option.as_s, option.as_t), tuple(deepcopy(trajectory)), tuple(deepcopy(action_trajectory))))
            trajectory.clear()
            action_trajectory.clear()
            if done:
                break

    return collect_option_trajectories.o_t

def static_collect_image_option_trajectories(env, option, render=True, o_t=list(), n_of_loops = 1):
    for i in range(n_of_loops):
        for init_state in option.init_set:
            state_trajectory = []
            action_trajectory = []
            env.reset()
            env.env.put_agent(init_state[0], init_state[1], 0)
            s, _, _, _ = env.step(4)                                            # perform a null action to get the observation WARNING WARNING WARNING WARNING
            state_trajectory.append(s["image"])
            if render:
                env.render()

            # plt.imshow()
            # plt.show(block=False)
            # plt.pause(0.001)

            while True:
                action = option.get_action(s["pos"])
                next_s, reward, done, info = env.step(action)
                if render:
                    env.render()
                    import time
                    time.sleep(1)

                state_trajectory.append(next_s["image"])
                action_trajectory.append(action)

                # end conditions
                s = next_s
                if done or option.done(next_s["pos"]):
                    break

            o_t.append(((option.as_s, option.as_t), tuple(deepcopy(state_trajectory)), tuple(deepcopy(action_trajectory)), None))
            state_trajectory.clear()
            action_trajectory.clear()

    return o_t

def static_collect_option_trajectories(env, option, render=True, o_t=list(), n_of_loops = 1):
    for i in range(n_of_loops):
        print("loop n:", i)
        for init_state in option.init_set:
            state_trajectory = []
            action_trajectory = []
            env.reset()
            env.env.put_agent(init_state[0], init_state[1], 0)
            s = init_state
            state_trajectory.append(s)
            if render:
                env.render()

            while True:
                action = option.get_action(s)
                next_s, reward, done, info = env.step(action)
                if render:
                    env.render()

                state_trajectory.append(next_s["pos"])
                action_trajectory.append(action)

                # end conditions
                s = next_s
                if done or option.done(next_s["pos"]):
                    break

            o_t.append(((option.as_s, option.as_t), tuple(deepcopy(state_trajectory)), tuple(deepcopy(action_trajectory)), None))
            state_trajectory.clear()
            action_trajectory.clear()

    return o_t

def xy_2_image(env, xy):
    env.reset()
    env.env.env.put_agent(xy[0], xy[1], 0)
    return env.env.env.get_grid_image()

def all_state_space_image(env, state_space_xy):
    all_state_space_img = []
    for state in state_space_xy:
        env.reset()
        env.env.env.put_agent(state[0], state[1], 0)
        s_image = env.env.env.get_grid_image()
        all_state_space_img.append(s_image)
    return all_state_space_img

def get_image_from_pos(env, x, render= False):
    np_array_img = np.zeros((env.env.width, env.env.height, 3), dtype=np.float32)
    goal = (14, 14)
    np_array_img[goal[0]][goal[1]] = (0, 1, 0)
    np_array_img[x[0]][x[1]] = (1, 0, 0)
    walls1 = set([(x, y) for x in [0] for y in (range(0, env.env.height))])
    walls2 = set([(y, x) for x in [0] for y in (range(0, env.env.height))])
    walls3 = set([(x, y) for x in [env.env.width-1] for y in (range(0, env.env.height))])
    walls4 = set([(y, x) for x in [env.env.width-1] for y in (range(0, env.env.height))])
    walls = walls1.union(walls2).union(walls3).union(walls4)
    for wall in walls:
        np_array_img[wall[0]][wall[1]] = (0.4, 0.4, 0.4)
    if render:
        plt.imshow(np_array_img)
        plt.show()
    return np_array_img

def pos_collect_option_trajectories(env, set_options, n_epochs=1000):
    all_state_visited = []
    for _ in range(n_epochs):
        trajectory = []
        action_trajectory = []
        s = env.reset()
        if s["pos"] not in all_state_visited:
            all_state_visited.append(s["pos"])
            pos_collect_option_trajectories.state_space.append(s["pos"])
        trajectory.append(s["pos"])
        while True:
            option = random.choice(set_options)
            while True:
                action = option.get_action(s)
                next_s, reward, done, info = env.step(action)
                #env.render()
                if next_s["pos"] not in all_state_visited:
                    all_state_visited.append(next_s["pos"])
                    pos_collect_option_trajectories.state_space.append(next_s["pos"])
                trajectory.append(next_s["pos"])
                action_trajectory.append(action)
                # end conditions
                s = next_s
                if done or option.done(next_s["pos"]):
                    break
            #if(len(trajectory)>2):   # WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
            pos_collect_option_trajectories.o_t.append(((option.as_s, option.as_t), tuple(deepcopy(trajectory)), tuple(deepcopy(action_trajectory)), option.id))
            trajectory.clear()
            action_trajectory.clear()
            trajectory.append(s["pos"])
            action_trajectory.append(action)
            if done:
                break

    return pos_collect_option_trajectories.o_t, pos_collect_option_trajectories.state_space

def image_collect_option_trajectories(env, set_options, n_epochs=1000):
    tot_n_steps_in_the_env = 0
    all_state_visited = []
    for _ in range(n_epochs):
        trajectory = []
        action_trajectory = []
        s = env.reset()
        if s["pos"] not in all_state_visited:
            all_state_visited.append(s["pos"])
            image_collect_option_trajectories.state_space.append(s["image"])
        trajectory.append(s["image"])
        while True:
            option = random.choice(set_options)
            while True:
                action = option.get_action(s)
                next_s, reward, done, info = env.step(action)
                tot_n_steps_in_the_env += 1
                #env.render()
                if next_s["pos"] not in all_state_visited:
                    all_state_visited.append(next_s["pos"])
                    image_collect_option_trajectories.state_space.append(next_s["image"])
                trajectory.append(next_s["image"])
                action_trajectory.append(action)
                # end conditions
                s = next_s
                if done or option.done(next_s["pos"]):
                    break
            #if(len(trajectory)>2):   # WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
            image_collect_option_trajectories.o_t.append(((option.as_s, option.as_t), tuple(deepcopy(trajectory)), tuple(deepcopy(action_trajectory)), option.id))
            trajectory.clear()
            action_trajectory.clear()
            trajectory.append(s["image"])
            action_trajectory.append(action)
            if done:
                break

    print("tot_n_steps_in_the_env: ", tot_n_steps_in_the_env)

    return image_collect_option_trajectories.o_t, image_collect_option_trajectories.state_space

def pos_collect_trajectories(env, set_options=None, n_epochs=1000):
    all_state_visited = []
    for _ in range(n_epochs):
        trajectory = []
        action_trajectory = []
        s = env.reset()
        if s["pos"] not in all_state_visited:
            all_state_visited.append(s["pos"])
            pos_collect_option_trajectories.state_space.append(s["pos"])
            pos_collect_option_trajectories.annotations.append(s["pos"])
        trajectory.append(s["pos"])
        while True:
            action = random.choice(list(range(env.action_space.n-1))) # We don't want null action 4
            next_s, reward, done, info = env.step(action)
            #env.render()
            if next_s["pos"] not in all_state_visited:
                all_state_visited.append(next_s["pos"])
                pos_collect_option_trajectories.state_space.append(next_s["pos"])
                pos_collect_option_trajectories.annotations.append(next_s["pos"])

            trajectory.append(next_s["pos"])
            action_trajectory.append(action)

            # end conditions
            if done:
                break

        pos_collect_option_trajectories.o_t.append(((None, None), tuple(deepcopy(trajectory)), tuple(deepcopy(action_trajectory)), None))
        trajectory.clear()
        action_trajectory.clear()

    return pos_collect_option_trajectories.o_t, pos_collect_option_trajectories.state_space, pos_collect_option_trajectories.annotations

def image_collect_trajectories(env, set_options=None, n_epochs=1000):
    tot_n_steps_in_the_env = 0
    all_state_visited = []
    for _ in tqdm.trange(n_epochs):
        trajectory = []
        action_trajectory = []
        s = env.reset()
        if s["pos"] not in all_state_visited:
            all_state_visited.append(s["pos"])
            image_collect_option_trajectories.state_space.append(s["image"])
            image_collect_option_trajectories.annotations.append(s["pos"])
        trajectory.append(s["image"])
        while True:
            action = random.choice(list(range(env.action_space.n-1))) # We don't want null action 4
            next_s, reward, done, info = env.step(action)
            tot_n_steps_in_the_env += 1
            #env.render()
            if next_s["pos"] not in all_state_visited:
                all_state_visited.append(next_s["pos"])
                image_collect_option_trajectories.state_space.append(next_s["image"])
                image_collect_option_trajectories.annotations.append(next_s["pos"])

            trajectory.append(next_s["image"])
            action_trajectory.append(action)

            # end conditions
            if done:
                break
        image_collect_option_trajectories.o_t.append(((None, None), tuple(deepcopy(trajectory)), tuple(deepcopy(action_trajectory)), None))
        trajectory.clear()
        action_trajectory.clear()

    print("tot_n_steps_in_the_env: ", tot_n_steps_in_the_env)

    return image_collect_option_trajectories.o_t, image_collect_option_trajectories.state_space, image_collect_option_trajectories.annotations

collect_transitions_and_reward.t_r = set()
collect_option_trajectories.o_t = list()
image_collect_option_trajectories.o_t = list()
image_collect_option_trajectories.state_space = list()
image_collect_option_trajectories.annotations = list()
pos_collect_option_trajectories.o_t = list()
pos_collect_option_trajectories.state_space = list()
pos_collect_option_trajectories.annotations = list()


def split_option_as_trajectories(trajectories):
    splitted_trajectory = []
    for trajectory in trajectories:
        for origin_as, final_as in zip(trajectory[0][0], trajectory[0][1]):
            splitted_trajectory.append(((origin_as, final_as), trajectory[1], trajectory[2], trajectory[3]))
    return splitted_trajectory

def split_sub_trajectories(trajectories, subset_split_div=1):
    splitted_trajectory = []
    for trajectory in trajectories:
        splitted_trajectory.append(trajectory)

        number_of_split = len(trajectory[1]) - 1
        random_split_trajectory = random.sample(list(range(1, number_of_split - subset_split_div)),
                                                int((number_of_split / subset_split_div) - 1))
        for i in random_split_trajectory:
            splitted_trajectory.append((trajectory[0], trajectory[1][i::], trajectory[2][i::], trajectory[3]))

    return splitted_trajectory


def combinatorial_split_option_as_trajectories(trajectories):
    splitted_trajectory = []
    for trajectory in trajectories:
        for origin_as in trajectory[0][0]:
            for final_as in trajectory[0][1]:
                splitted_trajectory.append(((origin_as, final_as), trajectory[1]))
    return splitted_trajectory


def abstract_grid_sets(number_of_abstract_states, env_w, env_h):   # only even sets supported for now
    env_w = env_w - 2  # walls
    env_h = env_h - 2  # walls
    number_of_abstract_states = number_of_abstract_states / 2
    list_of_sets = []
    row = []
    #all_state_space = set()
    quadrant = []

    id_as = -1

    if env_w % number_of_abstract_states == 0 and env_h % number_of_abstract_states == 0:
        n_s_as_w = int(env_w/number_of_abstract_states) # walls
        n_s_as_h = int(env_h/number_of_abstract_states) # walls

        for i in range(1, env_w+1, n_s_as_w):
            for j in range(1, env_h+1, n_s_as_h):
                id_as += 1
                for x in range(i, (i + n_s_as_w)):
                    for y in range(j, (j + n_s_as_h)):
                        quadrant.append((x, y))

                row.append((id_as, set(quadrant)))
                #all_state_space = all_state_space.union(quadrant)
                quadrant.clear()

            list_of_sets.append(deepcopy(row))
            row.clear()

        return list_of_sets#, all_state_space
    else:
        print("the size of the grid is not compatible with the discretization",
              env_w / number_of_abstract_states,
              env_h / number_of_abstract_states)
        breakpoint()

def sums(length, total_sum, divide):
    if length == 1:
        yield ((total_sum+1e-6)/divide,)   # to avoid 0 for log
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value, divide):
                yield (value+1e-6/divide,) + permutation  # to avoid 0 for log


def plot_to_tensorboard(writer, fig, step, name):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.transpose(img)
    img = np.swapaxes(img, 1, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image(name, img, step)
    plt.close(fig)

def plot_to_wandb(fig, step, name):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        wandb_writer (tensorboard.SummaryWriter): wandb instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Add figure in numpy "image" to TensorBoard writer
    wandb.log({name: [wandb.Image(img)]})
    plt.close(fig)

def NineRooms(config):
    assert "NineRoomsDet" in config["env"]

    if not config["load_data"]:
        env = gym.make(config["env"])
        env = RGBImgObsWrapper(env)
        env = NESWActionsImage(env, config["p_random_action"], config["max_len_episode"])

        print("loaded environment")

        # set seed
        env.seed(config["seed"])
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

        state_space_empty = env.env.env.grid.state_space_empty()
        state_space_empty_img = all_state_space_image(env, state_space_empty)
        state_space_lava = env.env.grid.state_space_w_lava()
        state_space_lava_img = all_state_space_image(env, state_space_lava)
        printable_state_space = np.array(state_space_empty)
        img_printable_state_space = np.array(state_space_empty_img)

        terminal_set_o1 = set()
        terminal_set_o2 = set()
        terminal_set_o3 = set()
        terminal_set_o4 = set()

        o1 = Option(set(state_space_empty), 1, terminal_set_o1, (0, 3, 6, 1, 4, 7, 2, 5, 8), (1, 4, 7, 2, 5, 8, 2, 5, 8))  # destra
        o2 = Option(set(state_space_empty), 2, terminal_set_o2, (0, 1, 2, 3, 4, 5, 6, 7, 8), (3, 4, 5, 6, 7, 8, 6, 7, 8))  # sopra
        o3 = Option(set(state_space_empty), 3, terminal_set_o3, (2, 5, 8, 1, 4, 7, 0, 3, 6), (1, 4, 7, 0, 3, 6, 0, 3, 6))  # sinistra
        o4 = Option(set(state_space_empty), 0, terminal_set_o4, (6, 7, 8, 3, 4, 5, 0, 1, 2), (3, 4, 5, 0, 1, 2, 0, 1, 2))  # sotto

        print("collecting data")

        if config["pos_or_image"] == "pos":
            dataset, dataset_states = pos_collect_option_trajectories(env, (o1, o2, o3, o4), n_epochs=config["n_episodes_env"])
        elif config["pos_or_image"] == "image":
            dataset, dataset_states = image_collect_option_trajectories(env, (o1, o2, o3, o4), n_epochs=config["n_episodes_env"])
        print("total_len_dataset =", len(dataset))

        env.render(close=True)
        env.reset()
        env.close()

        memory = ExperienceReplay(len(dataset))
        for trajectory in dataset:
            memory.add(torch.FloatTensor(trajectory[1]))

        filehandler = open("dataset", 'wb')
        pickle.dump(memory, filehandler)
        filehandler = open("dataset_states", 'wb')
        pickle.dump(dataset_states, filehandler)

    else:
        env = gym.make(config["env"])
        env = RGBImgObsWrapper(env)
        env = NESWActionsImage(env, config["p_random_action"], config["max_len_episode"])

        print("loaded environment")

        # set seed
        env.seed(config["seed"])
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

        state_space_empty = env.env.env.grid.state_space_empty()

        env.render(close=True)
        env.reset()
        env.close()

        filehandler = open("dataset", 'rb')
        memory = pickle.load(filehandler)
        filehandler = open("dataset_states", 'rb')
        dataset_states = pickle.load(filehandler)

    return memory, dataset_states, state_space_empty


def Random_traj_env(config):

    if not config["load_data"]:
        env = gym.make(config["env"])
        env = RGBImgObsWrapper(env)
        env = NESWActionsImage(env, config["p_random_action"], config["max_len_episode"])

        print("loaded environment")

        # set seed
        env.seed(config["seed"])
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

        print("collecting data")

        if config["pos_or_image"] == "pos":
            dataset, dataset_states, annotations = pos_collect_trajectories(env, None, n_epochs=config["n_episodes_env"])
        elif config["pos_or_image"] == "image":
            dataset, dataset_states, annotations = image_collect_trajectories(env, None, n_epochs=config["n_episodes_env"])
        print("total_len_dataset_trajectories =", len(dataset))

        env.reset()
        env.close()

        memory = ExperienceReplay_v2(len(dataset) * (len(dataset[0][1])-1))
        for trajectory in dataset:
            for i in range(len(trajectory[1])-1):
                memory.add((torch.FloatTensor(np.array((trajectory[1][i], trajectory[1][i+1]))), trajectory[2][i]))

        print("experience_replay_size =", len(memory.buffer))

        filehandler = open("memory", 'wb')
        pickle.dump(memory, filehandler)
        filehandler = open("dataset", 'wb')
        pickle.dump(dataset, filehandler)
        filehandler = open("dataset_states", 'wb')
        pickle.dump(dataset_states, filehandler)
        filehandler = open("annotations", 'wb')
        pickle.dump(annotations, filehandler)


    else:
        # set seed
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

        filehandler = open("memory", 'rb')
        memory = pickle.load(filehandler)
        filehandler = open("dataset", 'rb')
        dataset = pickle.load(filehandler)
        filehandler = open("dataset_states", 'rb')
        dataset_states = pickle.load(filehandler)
        filehandler = open("annotations", 'rb')
        annotations = pickle.load(filehandler)

    return memory, dataset, dataset_states, annotations


def single_plot(state_space, network, timeout=False):

    z = network.pred(torch.FloatTensor(state_space), 1).detach().numpy()
    argmax_z = np.argmax(z, axis=1)

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    fig, ax = plt.subplots()
    timer = fig.canvas.new_timer(interval=3000)  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)
    x = []
    y = []
    s = []
    for i, xy in enumerate(state_space):
        ax.annotate(np.around(z[i][argmax_z[i]], 2), xy)
        x.append(xy[0])
        y.append(xy[1])
        s.append(500)
    plt.scatter(x, y, c=argmax_z, s=s)
    if timeout:
        timer.start()
    plt.show()

def wandb_plot(state_space, annotation, network, d=2):
    lat_z = network.pred(torch.FloatTensor(state_space), 1).detach().numpy()
    argmax_z = np.argmax(lat_z, axis=1)

    x = []
    y = []
    z = []
    s = []
    if d == 2:
        fig, ax = plt.subplots()
        for i, xy in enumerate(annotation):
            xy = xy[0:2]
            #ax.annotate(str(argmax_z[i]) + " - " + str(np.around(lat_z[i][argmax_z[i]], 2)), xy)
            x.append(xy[0])
            y.append(xy[1])
            s.append(500)

    if d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i, xyz in enumerate(annotation):
            xyz = (xyz[0], xyz[1], xyz[2] + xyz[3])
            #ax.annotate(str(argmax_z[i]) + " - " + str(np.around(lat_z[i][argmax_z[i]], 2)), xyz)
            x.append(xyz[0])
            y.append(xyz[1])
            z.append(xyz[2])
            #s.append(500)

    if d == 2:
        plt.scatter(x, y, c=argmax_z, s=s)

    if d==3:
        ax.scatter(x, y, z, c=argmax_z, s=500)

    return fig

def representation_score(config, network):
    env = gym.make(config["env"])
    env = RGBImgObsWrapper(env)
    env = NESWActionsImage(env, 0, 100)

    network = network.eval()

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

    rooms = [room1, room2, room3, room4, room5, room6, room7, room8, room9]
    corridors = [corridor_room1_4, corridor_room1_2, corridor_room4_7, corridor_room4_5,
                 corridor_room2_5, corridor_room2_3, corridor_room5_8, corridor_room5_6,
                 corridor_room3_6, corridor_room6_9]

    identity = np.identity(9)
    a_s_templates = [tuple(x) for x in identity]

    cost = 0
    squared_cost = 0
    abs_cost = 0

    for i, room in enumerate(rooms):
        room_img = all_state_space_image(env, room)
        a_s_room = network.pred(torch.FloatTensor(room_img), 1).detach().numpy()
        count_a_s = np.zeros(9)
        for a_s_prob in a_s_room:
            a_s_argmax = np.argmax(a_s_prob)
            tmp = np.zeros(9)
            tmp[a_s_argmax] = 1
            count_a_s += tmp
        a_s = np.zeros(9)
        a_s[np.argmax(count_a_s)] = 1
        if tuple(a_s) in a_s_templates:
            a_s_templates.pop(a_s_templates.index(tuple(a_s)))
            for a_s_prob in a_s_room:
                cost += np.linalg.norm(a_s_prob-a_s)
                squared_cost += np.sum((a_s_prob-a_s)**2)
                abs_cost += np.sum(np.absolute((a_s_prob-a_s)))
        else:
            for a_s_prob in a_s_room:
                cost += np.linalg.norm(identity[0] - identity[1])
                squared_cost += np.sum((identity[0]-identity[1])**2)
                abs_cost += np.sum(np.absolute((identity[0]-identity[1])))
    env.close()

    return cost, squared_cost, abs_cost