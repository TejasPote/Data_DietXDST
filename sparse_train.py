import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from models import *
from data import *
from utils import *
import yaml
import os
import datetime
import argparse
import numpy as np 
import random 
import wandb




def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



def reinitialize_optimizer(optimizer, model, scheduler):

   optimizer.param_groups.clear()
   optimizer.add_param_group({'params': model.parameters(), 'lr':scheduler.get_last_lr()[0]})
   for name, param in model.state_dict().items():
        if "weight_orig" in name:
            # Get the corresponding mask
            mask = model.state_dict()[name.replace('_orig', '_mask')]
    
            # Calculate the pruned weight
            pruned_weight = param
            # Find the corresponding parameter in the optimizer
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.shape == pruned_weight.shape and torch.equal(p, pruned_weight):
                        # Update the optimizer state for this parameter
                        if p in optimizer.state:
                            if 'momentum_buffer' in optimizer.state[p]:
                                print("yes")
                                optimizer.state[p]['momentum_buffer'] *= mask




def main(args): 

    
    with open("sparse_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    current_time = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")


    wandb.init(
        project=config["project"],
        # entity=config["entity"],
        name=f"{config['model']['name']}; {config['training']['mode']}; {args.seed}",
        config={"data_sparsity": args.data_sparsity, "parameter_sparsity" : args.param_sparsity, 'seed': args.seed}
    )

    set_seed(args.seed)
    
    
    param_sparsity  = args.param_sparsity
    

    cwd = os.getcwd()

    # Initializing the sparse model with the same initialization as the dense model, 
    sparse_model = resnet20(config)
    init_ckpt = torch.load(cwd + f'/dense_checkpoints/seed_{args.seed}/init.pth')
    sparse_model.load_state_dict(init_ckpt, strict = True)
    
    
    optimizer_type = getattr(optim, config["optimizer"]["type"])

    

    optimizer = optimizer_type(
            sparse_model.parameters(),
            lr=config["optimizer"]["lr"],
            momentum=config["optimizer"]["momentum"],
            weight_decay=config["optimizer"]["weight_decay"],
            )   
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"],   eta_min=0.0)

    
    train_dl, test_dl, train_size = get_sparse_loader(sparse_model, config, args.data_sparsity)

    print(f"Size of pruned dataset : {train_size}")

    num_iters = len(train_dl)*config['epochs']
    if config['model']['rewind_frequency']:
        rewind_epochs = [i*config['model']['rewind_frequency'] for i in range(int(0.8*num_iters/config['model']['rewind_frequency']))]
    else:
        rewind_epochs = [0]

    save_folder = cwd + f"/sparse_ckpts/seed_{args.seed}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder + f"/ParamSparsity{args.param_sparsity}_DataSparsity_{args.data_sparsity}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Generating masks from checkpoints

    itr = 0
    for epoch in range(1, config['epochs']+1):
        running_loss = 0
        loss_fn = nn.CrossEntropyLoss()

        for _, X, y in train_dl:

            # Check if current training iteration is a multiple of rewind frequency
            if itr in rewind_epochs:

                # Load the dense model checkpoint from which sparse mask is to be generated  
                ckpt = resnet20(config)
                if itr == 0: 
                    
                    ckpt.load_state_dict(torch.load(cwd + f'/dense_checkpoints/seed_{args.seed}/init.pth'))
                    print(f"Loading initialization checkpoint...")

                else:
                    state_dict = torch.load(cwd + f'/dense_checkpoints/seed_{args.seed}/itr_{itr}.pth', weights_only = False)
                    ckpt.load_state_dict(state_dict["model"])
                
                # Calculate and print the sparsity of the model before changing the masks
                current_sparsity = calculate_overall_sparsity_from_pth(sparse_model)
                print(f"Current Sparsity : {current_sparsity}")    

                
                # Gather the layers to be pruned 
                print(f"Generating masks at epoch : {epoch}")

                parameters_to_prune = [
                    (module, 'weight') for name, module in ckpt.named_modules() 
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and 'shortcut' not in name
                ]

                # Before transferring the masks to the sparse model, it is necessary to remove the previous masks, else the masks get multiplied and sparsity increases
                if itr != 0:
                    for name, module in sparse_model.named_modules():
                        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and "shortcut" not in name: 
                            prune.remove(module, "weight")
                
                # Code to generate masks from the dense checkpoints, random pruning at iteration 0 and magnitude pruning otherwise.

                if itr == 0:
                    prune.global_unstructured(
                    parameters_to_prune, 
                    pruning_method=prune.RandomUnstructured, 
                    amount=param_sparsity
                    ) 
                else:
                    prune.global_unstructured(
                    parameters_to_prune, 
                    pruning_method=prune.L1Unstructured, 
                    amount=param_sparsity
                    )
                
                # The following block of code transfers the sparsity from the checkpoint to the sparse model
                transfer_sparsity_resnet(ckpt, sparse_model)

                # Since the masks change, it is necessary to reinitialize the optimizer and change the momentum buffers according to the new mask
                reinitialize_optimizer(optimizer, sparse_model, lrs)

                overall_sparsity = calculate_overall_sparsity_from_pth(sparse_model)
                print(f"Prune iteration: {itr} - Overall Parameter Sparsity: {overall_sparsity*100:.2f}%")
            

            # Sparse training schedule 
            
            sparse_model.train()
            X, y = X.to(config['device']), y.to(config['device'])
            optimizer.zero_grad()
            # Compute prediction error
            pred = sparse_model(X)
            loss = loss_fn(pred, y)
            running_loss += loss.item()
            itr += 1
            # Backpropagation
            loss.backward()
            optimizer.step()
           
            if itr%100 == 0:
                test_accu, test_loss = evaluate(sparse_model, test_dl, config['device'])
                state = {
                    "model": sparse_model,
                    "test_acc": test_accu
                }
                print(f'Saving at checkpoint {itr}')
                torch.save(state, save_path + f"/itr_{itr}.pth")


        
        
        
        train_loss_est = running_loss / len(train_dl)

        
        
                

        wandb.log(
                {
                    f"{config['model']['name']} {config['training']['mode']}  Epoch": epoch,
                    f"{config['model']['name']} {config['training']['mode']}  Train Loss": train_loss_est,
                    f"{config['model']['name']} {config['training']['mode']}  Test Loss": test_loss,
                    f"{config['model']['name']} {config['training']['mode']}  Test Accuracy": test_accu,
                    f"{config['model']['name']} {config['training']['mode']}  Lr": lrs.get_last_lr()[0],
                    f"{config['model']['name']} {config['training']['mode']} data sparsity": args.data_sparsity,
                    f"{config['model']['name']} {config['training']['mode']} sparsity": overall_sparsity,

                }
            )

        print(
            f"Epoch: {epoch:2d} - Test accuracy: {test_accu:.2f} - Test loss: {test_loss:.4f} - ",
            f"Train loss est.: {train_loss_est:.4f} - Learning rate: {optimizer.param_groups[0]['lr']:.4f}",
        )

        lrs.step()



if __name__ == "__main__":
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description="Train a Parameter and Data Sparse ResNet20 on CIFAR10")
    parser.add_argument("--seed", type=int, required=True, help="Random seed of the model")
    parser.add_argument("--param_sparsity", type=float, required=True, help="Desired sparsity in parameter space")
    parser.add_argument("--data_sparsity", type=float, required=True, help="Desired sparsity in data space")

    args = parser.parse_args()
    main(args)