from copy import deepcopy
import numpy as np
from soft_cluster import soft_option_replay_mdp

p_random_action = 0
env = "MiniGrid-NineRoomsDet-v0"
w1 = 1
w2 = 0.4
w3 = 0.1

config = {
    "env": env,
    "seed": 0,
    "batch_trajectories_size": 32,
    "epochs": 10000,
    "lr": 1e-4,
    "p_random_action": 0,
    "max_len_episode": 100,
    "path": "",
    "hidden_dim": 128,
    "n_abstract_states": 9,
    "print_interval": 100,
    "n_episodes_env": 1000,
    "pos_or_image": "image",
    "width": 19,
    "height": 19,
    "load_data": False,
    "percentual_of_trajectories": 1,
    "wandb_record": True,
    "wandb_group_name": "NineRoomDet",
    "wandb_project": "soft-options",
    "wandb_entity": "lsteccanella",  # change to your entity
    "plot_d": 3,
    "wl1": float(w1),
    "wl2": float(w2),
    "wl3": float(w3),
}

config["name_file"] = "ER_" + "weights" + str(w1) + "-" + str(w2) + "-" + str(w3) + "_percentage_traj"+ str(config["percentual_of_trajectories"])

list_errors = []
list_percentages_dataset = [1]
seeds = [0]
seed = None
list_error = []
list_squared_error = []
list_abs_error = []
list_seed = []
if __name__ == "__main__":
    for percentage in list_percentages_dataset:
        config["percentual_of_trajectories"] = percentage
        for s in seeds:
            seed = s
            config["seed"] = seed
            print("\nSEED: ", config["seed"], end="\n\n")
            config["name_file"] = "ER_" + str(env) + "_seed_" + str(seed) + "_weights" + str(w1) + "-" + str(w2) + "-" + str(
                w3) + "_percentage_traj" + str(config["percentual_of_trajectories"])

            error, squared_error, abs_error, n_traj, size_e = soft_option_replay_mdp(config)
            print("ERROR: ", error,
                  "\nSQUARED ERROR: ", squared_error,
                  "\nABS ERROR: ", abs_error,
                  "\nPERCENTAGE: ", percentage,
                  "\nN_TRAJ: ", n_traj,
                  "\nSIZE_E: ", size_e)
            list_error.append(error)
            list_squared_error.append(squared_error)
            list_abs_error.append(abs_error)
            list_seed.append(seed)
        list_errors.append((deepcopy(np.array(list_error)), deepcopy(np.array(list_squared_error)),
                            deepcopy(np.array(list_abs_error)), deepcopy(np.array(list_seed)), percentage,
                            n_traj, size_e))
        list_error.clear()
        list_squared_error.clear()
        list_abs_error.clear()
        list_seed.clear()

    # filehandler = open("list_errors", 'wb')
    # pickle.dump(list_errors, filehandler)
