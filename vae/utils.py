import os
import torch

def save_checkpoint(model, optimizer, training_state, path):
    if not os.path.exists(os.path.dirname(path)):
        print("Creating directories on path: `{}`".format(path))
        os.makedirs(os.path.dirname(path))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "latent_dim": model.latent_dim
        },
        "training_state": training_state
    }, path)

    print("Checkpoint saved to path: `{}`".format(path))

def load_checkpoint(path, map_location=None):
    """
    Load the checkpoint and return the dict. Do NOT instantiate the model/optimizer here.
    Caller is responsible for creating model and optimizer and loading states.

    Args:
        path: path to checkpoint
        map_location: device or None (if None, CPU by default via torch.load behavior)

    Returns:
        checkpoint (dict)
    """

    if map_location is None:
        # explicit safe default — load to CPU so the caller can move to the desired device
        map_location = torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint

    return torch.load(path) # loads on the device it was saved on

def get_optimal_device():
    """
    Return the best available device in order:
    - CUDA
    - MPS (Apple Silicon)
    - CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")