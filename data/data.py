import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn

class CustomDataset(Dataset):

    """
    A custom dataset that returns the index, image and label corresponding to each sample.
    Returning the indices of the samples will come handy in identifying the samples to prune.

    Args:
        Dataset : A dataset object that supports indexing (For ex. torchvision.Datasets)
    
    Returns:
        tuple: (index, image, label), where:
            - index is the sample index,
            - image is the transformed image tensor,
            - label is the corresponding class label.
    """
    def __init__(self, data):

        self.data = data 

    def __getitem__(self, idx):

        image = self.data[idx][0]
        label = self.data[idx][1]

        return idx, image, label
    
    def __len__(self):

        return len(self.data)
    
def load_cifar10(): 
    
    """
    Loads the CIFAR-10 dataset with predefined transformations and stores it in the home directory.

    Returns:
        tuple: (train_dataset, test_dataset)
            
    """

    
    print("Loading CIFAR10 dataset ...")
    
    # Define transforms to be applied to the train and test datasets 
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Use the CustomDataset class to load only the training data as it is the one which will be pruned. The test dataset is loaded as done conventionally using PyTorch.
    train_dataset = CustomDataset(torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=train_transform))
    test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=test_transform)
    
    return train_dataset, test_dataset

def get_dataloader(config):
    
    """
    Returns the dataloader for CIFAR-10 dataset.

    Args:
        config (dict): Configuration dictionary containing batch size and number of workers.

    Returns:
        tuple: (train_dataset, test_dataset)
            
    """
   

    # Fetch the batch size and num_workers for loading the dataset from the configuration file

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    
    # Load the train and test datasets 
    train_dataset, test_dataset = load_cifar10()
    train_samples = len(train_dataset)
    
    # Creating train and test loaders
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)

    return train_loader, test_loader, train_samples