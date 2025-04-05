import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models import *

from torch.nn.utils.prune import custom_from_mask
import math
import copy
import yaml
from collections import OrderedDict
from typing import Dict, Callable


def evaluate(model, dataloader, device):
    """Evaluate the model on the dataloader, return accuracy and loss. No TTA."""
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            
            pred = outputs.argmax(dim=1)
            correct += (y == pred).sum().item()
            total += y.size(0)
         
            loss = loss_fn(outputs, y)
            
            test_loss += loss.item()
            
    

    test_accuracy = correct / total
    test_loss /= len(dataloader)
    return test_accuracy * 100, test_loss




def evaluate(model, dataloader, device):
    """Evaluate the model on the dataloader, return accuracy and loss. No TTA."""
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            #print("Model outputs:", outputs, outputs.shape)
            pred = outputs.argmax(dim=1)
            correct += (y == pred).sum().item()
            total += y.size(0)
            #print(F.softmax(torch.tensor(outputs, dtype=torch.float64), dim=1)) 
            loss = loss_fn(outputs, y)
            #print("print X and y:", X.shape, y.dtype, y.shape)
            #print("Loss:", loss.item())
            test_loss += loss.item()
            
    #print("Last layer of the model:", list(model.children())[-1])

    test_accuracy = correct / total
    test_loss /= len(dataloader)
    return test_accuracy * 100, test_loss








def calculate_mask_sparsity(weight_mask):
    total_params = torch.numel(weight_mask)
    zero_params = torch.sum(weight_mask == 0).item()
    sparsity = zero_params / total_params
    return sparsity, zero_params, total_params

def calculate_overall_sparsity_from_pth(model):
    total_zero_params = 0
    total_params = 0

    for name, param in model.state_dict().items():
        if 'weight_mask' in name:
            _, zero_params, num_params = calculate_mask_sparsity(param)
            total_zero_params += zero_params
            total_params += num_params

    overall_sparsity = total_zero_params / total_params if total_params > 0 else 0
    return overall_sparsity


    


def transfer_sparsity_resnet(model_A, model_B):
    '''
    function to transfer sparsity for resnet model definition as per the REPAIR codebase
    args:
        model_A: sparse model with torch pruner mask/weight_orig
        model_B: dense model on which mask needs to applied
    return:
        modified model_B with mask applied in-place
    '''

    
    modules_to_replace = {}
    
    
    for (name_A, module_A), (name_B, module_B) in zip(model_A.named_modules(), model_B.named_modules()):
    
        # print(name_A, name_B)
    
        assert(type(module_A) is type(module_B) and name_A == name_B) # and hasattr(module_A, 'weight_mask'))
        
        if hasattr(module_A, "weight_mask"):
            print('Replacing layer in model B with masked layer:', name_A)
            modules_to_replace[name_A] = custom_from_mask(copy.deepcopy(module_B), name="weight", mask=module_A.weight_mask)
        else:
            print('Skipping layer in model A when copying masks:', name_A)
    
    # print(modules_to_replace)
    for module_name, module in modules_to_replace.items():
        with torch.no_grad():
            # print(module_name)
            mod_attr = model_B

            for i in range(len(module_name.split('.'))-1):
    
                if len(module_name.split('.')[i])!=1:
                    mod_attr = getattr(model_B, module_name.split('.')[i])
                else:
                    mod_attr = mod_attr[int(module_name.split('.')[i])]
                
                
    
            
            assert(hasattr(mod_attr, module_name.split('.')[-1]))
            setattr(mod_attr, module_name.split('.')[-1], module)
            # print(mod_attr, module_name)
            assert(hasattr(getattr(mod_attr, module_name.split('.')[-1]), 'weight_mask'))

                