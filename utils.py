import itertools
import os
import pickle
from copy import deepcopy
from typing import List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import tqdm
import wandb
from gym_minigrid.wrappers import *
import torch
import gym
from matplotlib.figure import Figure
from torch import nn
from numpy import ndarray
from replay import ExperienceReplay



# Define type aliases for better readability
StateType = Any  # Replace with a more specific type if possible (e.g., numpy.ndarray)
ActionType = int
TrajectoryType = Tuple[
    Tuple[Optional[StateType], Optional[StateType]],  # Option start and target states
    Tuple[StateType, ...],  # Sequence of states
    Tuple[ActionType, ...],  # Sequence of actions
    Optional[int],  # Option ID
]


def pos_collect_option_trajectories(env: gym.Env, set_options: List[Any], n_epochs: int = 1000) -> Tuple[List[TrajectoryType], List[StateType]]:
    """
    Collects trajectories by executing options in the environment, focusing on position data.

    Args:
        env: The Gym environment.
        set_options: A list of option objects.
        n_epochs: The number of epochs (episodes) to run.

    Returns:
        A tuple containing:
        - A list of trajectories, where each trajectory is a tuple containing option details, state sequence, action sequence, and option ID.
        - A list of all states visited during the data collection process.
    """
    all_state_visited: List[StateType] = []
    o_t: List[TrajectoryType] = []
    state_space: List[StateType] = []

    for _ in range(n_epochs):
        trajectory: List[StateType] = []
        action_trajectory: List[ActionType] = []
        s = env.reset()

        # Add initial state
        if s["pos"] not in all_state_visited:
            all_state_visited.append(s["pos"])
            state_space.append(s["pos"])
        trajectory.append(s["pos"])

        while True:
            option = random.choice(set_options)
            while True:
                action = option.get_action(s)
                next_s, reward, done, info = env.step(action)
                #env.render()  # Consider commenting out rendering for performance

                # Add new state
                if next_s["pos"] not in all_state_visited:
                    all_state_visited.append(next_s["pos"])
                    state_space.append(next_s["pos"])
                trajectory.append(next_s["pos"])
                action_trajectory.append(action)

                s = next_s
                if done or option.done(next_s["pos"]):
                    break

            o_t.append(
                ((option.as_s, option.as_t), tuple(deepcopy(trajectory)), tuple(deepcopy(action_trajectory)), option.id)
            )

            trajectory.clear()
            action_trajectory.clear()
            trajectory.append(s["pos"])
            action_trajectory.append(action)

            if done:
                break

    return o_t, state_space


def image_collect_option_trajectories(env: gym.Env, set_options: List[Any], n_epochs: int = 1000) -> Tuple[List[TrajectoryType], List[StateType]]:
    """
    Collects trajectories by executing options in the environment, focusing on image data.

    Args:
        env: The Gym environment.
        set_options: A list of option objects.
        n_epochs: The number of epochs (episodes) to run.

    Returns:
        A tuple containing:
        - A list of trajectories, where each trajectory is a tuple containing option details, state sequence, action sequence, and option ID.
        - A list of all states visited during the data collection process.
    """
    tot_n_steps_in_the_env = 0
    all_state_visited: List[StateType] = []
    o_t: List[TrajectoryType] = []
    state_space: List[StateType] = []

    for _ in range(n_epochs):
        trajectory: List[StateType] = []
        action_trajectory: List[ActionType] = []
        s = env.reset()

        # Add initial state
        if s["pos"] not in all_state_visited:
            all_state_visited.append(s["pos"])
            state_space.append(s["image"])
        trajectory.append(s["image"])

        while True:
            option = random.choice(set_options)
            while True:
                action = option.get_action(s)
                next_s, reward, done, info = env.step(action)
                tot_n_steps_in_the_env += 1
                #env.render()  # Consider commenting out rendering for performance

                # Add new state
                if next_s["pos"] not in all_state_visited:
                    all_state_visited.append(next_s["pos"])
                    state_space.append(next_s["image"])
                trajectory.append(next_s["image"])
                action_trajectory.append(action)

                s = next_s
                if done or option.done(next_s["pos"]):
                    break

            o_t.append(
                ((option.as_s, option.as_t), tuple(deepcopy(trajectory)), tuple(deepcopy(action_trajectory)), option.id)
            )

            trajectory.clear()
            action_trajectory.clear()
            trajectory.append(s["image"])
            action_trajectory.append(action)

            if done:
                break

    print("tot_n_steps_in_the_env: ", tot_n_steps_in_the_env)

    return o_t, state_space


def pos_collect_trajectories(env: gym.Env, n_epochs: int = 1000) -> Tuple[List[TrajectoryType], List[StateType], List[StateType]]:
    """Collects trajectories of positions from the environment with random actions.

    Args:
        env: The Gym environment.
        n_epochs: The number of epochs (episodes) to run.

    Returns:
        A tuple containing:
            - A list of trajectories, where each trajectory contains state sequence, action sequence, and option ID.
            - A list of all states visited.
            - A list of annotations for the states.
    """
    all_state_visited: List[StateType] = []
    o_t: List[TrajectoryType] = []
    state_space: List[StateType] = []
    annotations: List[StateType] = []

    for _ in range(n_epochs):
        trajectory: List[StateType] = []
        action_trajectory: List[ActionType] = []
        s = env.reset()

        if s["pos"] not in all_state_visited:
            all_state_visited.append(s["pos"])
            state_space.append(s["pos"])
            annotations.append(s["pos"])
        trajectory.append(s["pos"])

        while True:
            action = random.choice(list(range(env.action_space.n - 1)))  # Exclude null action 4
            next_s, reward, done, info = env.step(action)

            if next_s["pos"] not in all_state_visited:
                all_state_visited.append(next_s["pos"])
                state_space.append(next_s["pos"])
                annotations.append(next_s["pos"])

            trajectory.append(next_s["pos"])
            action_trajectory.append(action)

            if done:
                break

        o_t.append(((None, None), tuple(deepcopy(trajectory)), tuple(deepcopy(action_trajectory)), None))

    return o_t, state_space, annotations


def image_collect_trajectories(env: gym.Env, n_epochs: int = 1000) -> Tuple[List[TrajectoryType], List[StateType], List[StateType]]:
    """Collects trajectories of images from the environment with random actions.

    Args:
        env: The Gym environment.
        n_epochs: The number of epochs (episodes) to run.

    Returns:
        A tuple containing:
            - A list of trajectories, where each trajectory contains state sequence, action sequence, and option ID.
            - A list of all states visited.
            - A list of annotations for the states.
    """
    tot_n_steps_in_the_env = 0
    all_state_visited: List[StateType] = []
    o_t: List[TrajectoryType] = []
    state_space: List[StateType] = []
    annotations: List[StateType] = []

    for _ in tqdm.trange(n_epochs):
        trajectory: List[StateType] = []
        action_trajectory: List[ActionType] = []
        s = env.reset()

        if s["pos"] not in all_state_visited:
            all_state_visited.append(s["pos"])
            state_space.append(s["image"])
            annotations.append(s["pos"])
        trajectory.append(s["image"])

        while True:
            action = random.choice(list(range(env.action_space.n - 1)))  # Exclude null action 4
            next_s, reward, done, info = env.step(action)
            tot_n_steps_in_the_env += 1

            if next_s["pos"] not in all_state_visited:
                all_state_visited.append(next_s["pos"])
                state_space.append(next_s["image"])
                annotations.append(next_s["pos"])

            trajectory.append(next_s["image"])
            action_trajectory.append(action)

            if done:
                break

        o_t.append(((None, None), tuple(deepcopy(trajectory)), tuple(deepcopy(action_trajectory)), None))

    print("tot_n_steps_in_the_env: ", tot_n_steps_in_the_env)

    return o_t, state_space, annotations


def plot_to_wandb(fig: Figure, step: int, name: str) -> None:
    """Converts a Matplotlib figure to a W&B image and logs it.

    Args:
        fig: The Matplotlib figure to log.
        step: The current step (e.g., epoch number).
        name: The name to use for the W&B image.
    """
    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Add figure in numpy "image" to TensorBoard writer
    wandb.log({name: [wandb.Image(img)]})
    plt.close(fig)


def collect_trajectories(config: dict) -> Tuple[Any, List[TrajectoryType], List[StateType], List[StateType]]:
    """Initializes the environment and collects trajectories based on the configuration.

    Args:
        config: A dictionary containing configuration parameters.

    Returns:
        A tuple containing:
            - The replay buffer.
            - The dataset of trajectories.
            - The list of states.
            - The list of annotations.
    """

    data_dir = "data"  # Define the data directory
    os.makedirs(data_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define file paths
    memory_path = os.path.join(data_dir, "memory.pkl")
    dataset_path = os.path.join(data_dir, "dataset.pkl")
    dataset_states_path = os.path.join(data_dir, "dataset_states.pkl")
    annotations_path = os.path.join(data_dir, "annotations.pkl")

    if config["load_data"]:
        # Set seed
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

        # Load data
        try:
            with open(memory_path, 'rb') as filehandler:
                memory = pickle.load(filehandler)
            with open(dataset_path, 'rb') as filehandler:
                dataset = pickle.load(filehandler)
            with open(dataset_states_path, 'rb') as filehandler:
                dataset_states = pickle.load(filehandler)
            with open(annotations_path, 'rb') as filehandler:
                annotations = pickle.load(filehandler)
        except FileNotFoundError:
            print("One or more data files not found. Ensure they exist or set load_data to False.")
            raise  # Re-raise the exception to halt execution

    else:
        env = gym.make(config["env"])
        env = RGBImgObsWrapper(env)
        env = NESWActionsImage(env, config["p_random_action"], config["max_len_episode"])

        print("loaded environment")

        # Set seed
        env.seed(config["seed"])
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

        print("collecting data")

        if config["pos_or_image"] == "pos":
            dataset, dataset_states, annotations = pos_collect_trajectories(env, n_epochs=config["n_episodes_env"])
        elif config["pos_or_image"] == "image":
            dataset, dataset_states, annotations = image_collect_trajectories(env, n_epochs=config["n_episodes_env"])
        print("total_len_dataset_trajectories =", len(dataset))

        env.reset()
        env.close()

        memory = ExperienceReplay(len(dataset) * (len(dataset[0][1]) - 1))
        for trajectory in dataset:
            for i in range(len(trajectory[1]) - 1):
                memory.add((torch.FloatTensor(np.array((trajectory[1][i], trajectory[1][i + 1]))), trajectory[2][i]))

        print("experience_replay_size =", len(memory.buffer))

        # Save data
        with open(memory_path, 'wb') as filehandler:
            pickle.dump(memory, filehandler)
        with open(dataset_path, 'wb') as filehandler:
            pickle.dump(dataset, filehandler)
        with open(dataset_states_path, 'wb') as filehandler:
            pickle.dump(dataset_states, filehandler)
        with open(annotations_path, 'wb') as filehandler:
            pickle.dump(annotations, filehandler)

    return memory, dataset, dataset_states, annotations


def single_plot(state_space: List[StateType], network: nn.Module, timeout: bool = False) -> None:
    """Plots the state space with cluster assignments from the network.

    Args:
        state_space: A list of states to plot.
        network: The trained neural network.
        timeout: Whether to close the plot window after a timeout.
    """
    z: ndarray = network.pred(torch.FloatTensor(state_space), 1).detach().numpy()
    argmax_z: ndarray = np.argmax(z, axis=1)

    def close_event():
        plt.close()

    fig, ax = plt.subplots()
    if timeout:
        timer = fig.canvas.new_timer(interval=3000)
        timer.add_callback(close_event)
        timer.start()

    x: List[float] = []
    y: List[float] = []
    s: List[int] = []
    for i, xy in enumerate(state_space):
        ax.annotate(np.around(z[i][argmax_z[i]], 2), xy)
        x.append(xy[0])
        y.append(xy[1])
        s.append(500)
    plt.scatter(x, y, c=argmax_z, s=s)
    plt.show()


def wandb_plot(state_space: List[StateType], annotation: List[StateType], network: nn.Module, d: int = 2) -> Figure:
    """Creates a scatter plot of the state space with cluster assignments and logs it to W&B.

    Args:
        state_space: A list of states.
        annotation: A list of annotations for the states.
        network: The trained neural network.
        d: The number of dimensions to plot (2 or 3).

    Returns:
        The Matplotlib figure.
    """
    # Move data to CPU for plotting
    device = next(network.parameters()).device
    state_space = np.array(state_space)
    state_space_tensor = torch.FloatTensor(state_space).to(device)
    lat_z: ndarray = network.pred(state_space_tensor, 1).detach().cpu().numpy() # Move the tensor to CPU before converting to NumPy
    argmax_z: ndarray = np.argmax(lat_z, axis=1)

    x: List[float] = []
    y: List[float] = []
    z_coords: List[float] = []
    s: List[int] = []

    if d == 2:
        fig, ax = plt.subplots()
        for i, xy in enumerate(annotation):
            xy = xy[0:2]
            #ax.annotate(str(argmax_z[i]) + " - " + str(np.around(lat_z[i][argmax_z[i]], 2)), xy)
            x.append(xy[0])
            y.append(xy[1])
            s.append(500)
        plt.scatter(x, y, c=argmax_z, s=s)

    elif d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i, xyz in enumerate(annotation):
            xyz = (xyz[0], xyz[1], xyz[2] + xyz[3])
            #ax.annotate(str(argmax_z[i]) + " - " + str(np.around(lat_z[i][argmax_z[i]], 2)), xyz)
            x.append(xyz[0])
            y.append(xyz[1])
            z_coords.append(xyz[2])
            #s.append(500)
        ax.scatter(x, y, z_coords, c=argmax_z, s=500)
    else:
        raise ValueError("d must be 2 or 3")

    return fig


def all_state_space_image(env, positions: List[Tuple[int, int]]) -> List[np.ndarray]:
    """Generates image representations for a list of positions in the environment.

    Args:
        env: The Gym environment.
        positions: A list of (x, y) positions.

    Returns:
        A list of image arrays.
    """
    images: List[np.ndarray] = []
    for pos in positions:
        env.reset()
        env.env.env.put_agent(pos[0], pos[1], 0)
        image: np.ndarray = env.env.env.get_grid_image()
        images.append(image)
    return images


def representation_score(config: dict, network: nn.Module) -> Tuple[float, float, float]:
    """Calculates a representation score based on the network's ability to cluster states.

    Args:
        config: A dictionary containing configuration parameters.
        network: The trained neural network.

    Returns:
        A tuple containing the total cost, squared cost, and absolute cost.
    """
    env = gym.make(config["env"])
    env = RGBImgObsWrapper(env)
    env = NESWActionsImage(env, 0, 100)

    network = network.eval()  # Set the network to evaluation mode

    # Define room coordinates
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

    # Define corridor coordinates
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

    cost: float = 0.0
    squared_cost: float = 0.0
    abs_cost: float = 0.0

    # get the device of the model
    device = next(network.parameters()).device

    for i, room in enumerate(rooms):
        room_img: List[np.ndarray] = all_state_space_image(env, room)
        room_img = np.array(room_img)
        room_img_tensor = torch.FloatTensor(room_img).to(device) # Shape: (num_positions, C, H, W)

        a_s_room: np.ndarray = network.pred(room_img_tensor, 1).detach().cpu().numpy() # Move the tensor to CPU before converting to NumPy
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
                cost += np.linalg.norm(a_s_prob - a_s)
                squared_cost += np.sum((a_s_prob - a_s)**2)
                abs_cost += np.sum(np.absolute((a_s_prob - a_s)))
        else:
            for a_s_prob in a_s_room:
                cost += np.linalg.norm(identity[0] - identity[1])
                squared_cost += np.sum((identity[0] - identity[1])**2)
                abs_cost += np.sum(np.absolute((identity[0] - identity[1])))
    env.close()

    return cost, squared_cost, abs_cost