import torch
import torchvision
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from .data import *  



def compute_scores(model, device, train_loader):
    """
    Computes the L2 norm of the error between model predictions and one-hot encoded targets.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        device (torch.device): Device to perform computations (CPU/GPU).
        train_loader (DataLoader): DataLoader for training dataset.

    Returns:
        dict: A dictionary mapping each sample index to its computed L2 norm error score.
    """
       
    scores = {}  # Dictionary to store computed scores per index
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)
    
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for batch_idx, (idx, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the correct device
            
            outputs = model(inputs)  # Forward pass through the model
            preds = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities

            # Compute one-hot encoded target vector
            one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=10).to(device)
            
            # Compute the L2 norm of the prediction error
            error = preds - one_hot_targets  # Compute prediction error
            el2n = error.norm(dim=1, p=2)  # Compute L2 norm (Euclidean distance)

            # Store the computed score for each index in the dataset
            for i, score in zip(idx, el2n):
                scores[i.item()] = score.item()

    return scores  # Return dictionary of computed scores

def get_mean_scores(model, device, train_loader, checkpoint_dir, iteration):
    """
    Computes the mean EL2N score across multiple checkpoints (from different random seeds).

    Args:
        model (torch.nn.Module): Model used for evaluation.
        device (torch.device): Device for computation.
        train_loader (DataLoader): DataLoader for training data.
        checkpoint_dir (str): Directory containing checkpoints for different seeds.
        iteration (int): The iteration number of the checkpoint to use.

    Returns:
        dict: Dictionary mapping each data index to its mean EL2N score.
    """
    scores_dict = {}  # Dictionary to store scores across different seeds

    # Iterate through all seed directories inside checkpoint_dir
    for seed_dir in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, seed_dir, f"itr_{iteration}.pth")

        # Check if the checkpoint file exists
        if os.path.exists(checkpoint_path):
            # Load the model checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])  # Load model weights
            
            # Compute scores for the current checkpoint
            scores = compute_scores(model, device, train_loader)

            # Accumulate scores across different checkpoints
            for idx, score in scores.items():
                if idx not in scores_dict:
                    scores_dict[idx] = []
                scores_dict[idx].append(score)

    # Compute the mean score for each data index
    mean_scores = {idx: np.mean(scores) for idx, scores in scores_dict.items() if scores}
    
    return mean_scores  # Return dictionary of mean scores


def prune_dataset(train_dataset, mean_scores, sparsity):

    """
    Prunes the dataset by keeping only a subset of samples based on EL2N scores.

    Args:
        train_dataset (Dataset): The full training dataset.
        mean_scores (dict): Dictionary of mean EL2N scores per sample index.
        sparsity (float): Fraction of the dataset to remove (e.g., 0.2 removes 20%).

    Returns:
        Subset: A pruned dataset containing only the most informative samples.
    """

    train_samples = len(train_dataset)  # Total number of training samples
    samples_to_keep = int((1 - sparsity) * train_samples)  # Compute number of samples to retain

    # Sort indices by score in descending order (higher scores are more informative)
    sorted_indices = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:samples_to_keep]

    # Extract only the indices from the sorted list
    selected_indices = [idx for idx, _ in sorted_indices]

    # Create a subset of the original dataset using selected indices
    pruned_dataset = Subset(train_dataset, selected_indices)

    print(f"Pruned dataset size: {len(pruned_dataset)}")  # Print new dataset size
    
    return pruned_dataset  # Return pruned dataset

    
    

def get_sparse_loader(model, config, data_sparsity, checkpoint_dir='dense_checkpoints'):
    """
    Loads the CIFAR-10 dataset, computes EL2N scores, prunes the dataset, 
    and returns DataLoaders for sparse training and testing.

    Args:
        model (torch.nn.Module): The neural network model.
        config (dict): Configuration dictionary containing training parameters.
        data_sparsity (float): Fraction of training data to remove.
        checkpoint_dir (str, optional): Directory where model checkpoints are stored. Defaults to 'dense_checkpoints'.

    Returns:
        tuple: (sparse_train_loader, test_loader, new dataset size)
            - sparse_train_loader: DataLoader for pruned training dataset.
            - test_loader: DataLoader for test dataset.
            - len(pruned_dataset): Number of samples remaining after pruning.
    """
    # Load the CIFAR-10 dataset (train & test)
    train_dataset, test_dataset = load_cifar10()

    # Create DataLoader for the full training dataset
    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"], 
        num_workers=config["data"]["num_workers"], shuffle=True, 
        drop_last=True, pin_memory=True
    )

    # Create DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset, batch_size=config["data"]["batch_size"], 
        num_workers=config["data"]["num_workers"], shuffle=False, 
        drop_last=False, pin_memory=True
    )

    train_samples = len(train_dataset)  # Get the total number of training samples

    # Compute the mean EL2N scores using multiple checkpoints
    mean_scores = get_mean_scores(model, config["device"], train_loader, checkpoint_dir, config["data"]["iteration"])

    # Prune the training dataset based on computed mean scores
    pruned_dataset = prune_dataset(train_dataset, mean_scores, data_sparsity)

    # Create a DataLoader for the pruned training dataset
    sparse_train_loader = DataLoader(
        pruned_dataset, batch_size=config["data"]["batch_size"], 
        shuffle=True, num_workers=config["data"]["num_workers"]
    )

    return sparse_train_loader, test_loader, len(pruned_dataset)  # Return pruned dataset loader and test loader

