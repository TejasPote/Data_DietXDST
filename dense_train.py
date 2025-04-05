import copy
import torch
import torch.nn as nn
import torch.optim as optim
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


def main(args): 

    
    with open("dense_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    current_time = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")


    wandb.init(
        project=config["project"],
        # entity=config["entity"],
        name=f"{config['model']['name']}; {config['training']['mode']}; {args.seed}",
        config={'seed': args.seed}
    )

    set_seed(args.seed)
    
    save_path = os.path.join(os.getcwd(), f"dense_checkpoints/seed_{args.seed}")
    os.makedirs(save_path, exist_ok=True)  # Ensure the checkpoint directory exists


    model = resnet20(config)
    torch.save(model.state_dict(), save_path + "/init.pth")
    

    
    
    optimizer_type = getattr(optim, config["optimizer"]["type"])

    

    optimizer = optimizer_type(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            momentum=config["optimizer"]["momentum"],
            weight_decay=config["optimizer"]["weight_decay"],
            )   
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=0.0)
    
    train_dl, test_dl, train_samples = get_dataloader(config)
    
    itr = 0
    
    
    for epoch in range(1, config['epochs']+1):
        running_loss = 0
        loss_fn = nn.CrossEntropyLoss()

        for indices, X, y in train_dl:

            model.train()
            X, y = X.to(config['device']), y.to(config['device'])
            optimizer.zero_grad()
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            running_loss += loss.item()
            itr += 1
            # Backpropagation
            loss.backward()
            optimizer.step()
           
            if itr%100 == 0:
                test_accu, test_loss = evaluate(model, test_dl, config['device'])
                state = {
                    "model": model.state_dict(),
                    "test_acc": test_accu
                }
                print(f'Saving at checkpoint {itr}')
                torch.save(state, save_path + f"/itr_{itr}.pth")

            

        train_loss_est = running_loss / len(train_dl)
        test_accu, test_loss = evaluate(model, test_dl, config['device'])

        wandb.log(
                {
                    f"{config['model']['name']} {config['training']['mode']}  Epoch": epoch,
                    f"{config['model']['name']} {config['training']['mode']}  Train Loss": train_loss_est,
                    f"{config['model']['name']} {config['training']['mode']}  Test Loss": test_loss,
                    f"{config['model']['name']} {config['training']['mode']}  Test Accuracy": test_accu,
                    f"{config['model']['name']} {config['training']['mode']}  Lr": lrs.get_last_lr()[0],

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
    parser = argparse.ArgumentParser(description="Train Dense ResNet20 on CIFAR10")
    parser.add_argument("--seed", type=int, required=True, help="Random seed of the model")
    args = parser.parse_args()
    main(args)