import random
from typing import Dict, Tuple, Any
import torch
import torch.optim as optim
import wandb
from replay import ExperienceReplay
from utils import collect_trajectories, wandb_plot, plot_to_wandb, representation_score
from model import SoftClusterNetwork


def soft_option_replay_mdp(config: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
    """
    Trains a SoftClusterNetwork using experience replay and evaluates its representation.

    Args:
        config: A dictionary containing configuration parameters.

    Returns:
        A tuple containing the error, squared error, absolute error, number of trajectories, and replay buffer size.
    """

    # Initialize Weights & Biases (wandb) if enabled
    if config["wandb_record"]:
        run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            config=config,
            group=config["wandb_group_name"],
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.run.name = config["path"] + config["name_file"]

    # Prepare environment and data
    memory, trajectories_dataset, print_states, annotations = collect_trajectories(config)

    # Subset the dataset if requested
    n_trajectories = int(len(trajectories_dataset) * config["percentual_of_trajectories"])
    print("n of trajectories: ", n_trajectories)

    if config["percentual_of_trajectories"] < 1:
        random.shuffle(trajectories_dataset)
        memory = ExperienceReplay(n_trajectories * (len(trajectories_dataset[0][1]) - 1))
        for trajectory in trajectories_dataset[0:n_trajectories]:
            for i in range(len(trajectory[1]) - 1):
                memory.add((torch.FloatTensor((trajectory[1][i], trajectory[1][i + 1])), trajectory[2][i]))

        print("experience_replay_size =", len(memory.buffer))

    # Initialize the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine device
    network = SoftClusterNetwork(config["n_abstract_states"], config["width"], config["height"], device=device)  # Pass device to model
    network.train()  # Set network to training mode

    # Track network gradients in wandb
    if config["wandb_record"]:
        wandb.watch(network)

    # Initialize the optimizer
    optimizer = optim.AdamW(network.parameters(), lr=config["lr"])

    # Training loop
    PRINT = config["print_interval"]
    print("Training")

    for i in range(config["epochs"]):
        # Sample a batch from the replay buffer
        idx, batch_x1, batch_x2, b_is_weights = memory.sample(config["batch_trajectories_size"])

        # Prepare input tensors and move them to the device
        x1 = torch.stack(batch_x1).to(device)
        x2 = torch.stack(batch_x2).to(device)

        # Calculate abstract states
        z1 = network.pred(x1, 1)
        z2 = network.pred(x2, 1)

        # Calculate losses
        compression_loss = ((-(z1 * z2.log())).sum(axis=1)).mean(axis=0) / config["batch_trajectories_size"]
        z = z1
        z_mean = z.mean(dim=0)
        entropy_loss = (z_mean * (z_mean.log())).sum()
        det_entropy_loss = (- (z * z.log()).sum(dim=1)).mean()

        loss = config["wl1"] * compression_loss + config["wl2"] * entropy_loss + config["wl3"] * det_entropy_loss

        # Log training information to wandb
        if config["wandb_record"] and i % 10 == 0:
            loss_info = {
                "compression_loss": compression_loss,
                "entropy_loss": entropy_loss,
                "det_entropy_loss": det_entropy_loss
            }
            for k, v in loss_info.items():
                wandb.log({f"train/{k}": v}, step=i)

        # Print training progress
        if i % PRINT == 0:
            print(i, " loss: ", loss.item())  # Use loss.item() to get the scalar value

        # Visualize progress and save model weights
        if i % PRINT == 0 and config["wandb_record"]:
            fig = wandb_plot(print_states, annotations, network, d=config["plot_d"])
            plot_to_wandb(fig, i, 'grid_assignment_scalar')
            network.save(wandb.run.dir + "/weights")

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        for param in network.parameters():
            param.grad.data.clamp_(-1, 1)  # Gradient clipping
        optimizer.step()

    # Save final model weights
    if config["wandb_record"]:
        network.save(wandb.run.dir + "/weights")
        run.finish()

    # Evaluate the learned representation
    error, squared_error, abs_error = representation_score(config, network)

    return error, squared_error, abs_error, n_trajectories, memory.buffer_len()