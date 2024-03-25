import math
import random

import torch
import torch.optim as optim

from replay import ExperienceReplay_v2
from utils import Random_traj_env, single_plot, wandb_plot, plot_to_wandb, representation_score
from model import SoftClusterNetwork
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16, 8)
import wandb

def soft_option_replay_mdp(config):
    # 1. Start a new run
    if(config["wandb_record"]):
        run = wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config, group=config["wandb_group_name"],
                         settings=wandb.Settings(start_method="fork"))
        wandb.run.name = config["path"] + config["name_file"]

    memory, trajectories_dataset, print_states, annotations = Random_traj_env(config)

    print("n of trajectories: ", int(len(trajectories_dataset) * config["percentual_of_trajectories"]))

    if config["percentual_of_trajectories"] < 1:
        random.shuffle(trajectories_dataset)
        n_trajectories = int(len(trajectories_dataset) * config["percentual_of_trajectories"])

        print("original n of trajectories ", len(trajectories_dataset), " new number of trajectories ", n_trajectories)

        memory = ExperienceReplay_v2(n_trajectories * (len(trajectories_dataset[0][1]) - 1))
        for trajectory in trajectories_dataset[0:n_trajectories]:
            for i in range(len(trajectory[1]) - 1):
                memory.add((torch.FloatTensor((trajectory[1][i], trajectory[1][i + 1])), trajectory[2][i]))

        print("experience_replay_size =", len(memory.buffer))

    # # # Training the network
    network = SoftClusterNetwork(config["n_abstract_states"], config["width"], config["height"])
    network.train()

    if(config["wandb_record"]):
        wandb.watch(network)

    optimizer = optim.AdamW(network.parameters(), lr=config["lr"])

    PRINT = config["print_interval"]

    print("Training")

    for i in range(config["epochs"]):

        idx, batch_x1, batch_x2, b_is_weights = memory.sample(config["batch_trajectories_size"])
        compression_loss = 0
        entropy_loss = 0
        det_entropy_loss = 0

        x1 = torch.stack(batch_x1)
        x2 = torch.stack(batch_x2)
        z1 = network.pred(x1, 1)
        z2 = network.pred(x2, 1)
        compression_loss += ((-(z1 * z2.log())).sum(axis=1)).mean(axis=0)
        compression_loss = compression_loss / config["batch_trajectories_size"]
        z = z1
        z_mean = z.mean(dim=0)
        entropy_loss += (z_mean * (z_mean.log())).sum()
        det_entropy_loss += (- (z * z.log()).sum(dim=1)).mean()

        loss = config["wl1"] * compression_loss + config["wl2"] * entropy_loss + config["wl3"] * det_entropy_loss
        loss = loss

        if (config["wandb_record"]) and i % 10 == 0:
            loss_info = {
                "compression_loss": compression_loss,
                "entropy_loss": entropy_loss,
                "det_entropy_loss": det_entropy_loss
            }
            for k, v in loss_info.items():
                wandb.log({f"train/{k}": v}, step=i)

        if i % PRINT == 0:
            print(i, " loss: ", loss)

        if i % PRINT == 0 and (config["wandb_record"]):
            fig = wandb_plot(print_states, annotations, network, d=config["plot_d"])
            plot_to_wandb(fig, i, 'grid_assignment_scalar')
            network.save(wandb.run.dir + "/weights")

        optimizer.zero_grad()
        loss.backward()
        for param in network.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    if (config["wandb_record"]):
        network.save(wandb.run.dir + "/weights")
        run.finish()

    error, squared_error, abs_error = representation_score(config, network)

    return error, squared_error, abs_error, int(len(trajectories_dataset) * config["percentual_of_trajectories"]), memory.buffer_len()


