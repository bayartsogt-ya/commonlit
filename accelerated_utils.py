import os
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True


def eval_mse(model, data_loader, device, accelerator):
    """Evaluates the mean squared error of the |model| on |data_loader|"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, target) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)                        
            target = target.to(device)           
            
            pred = model(input_ids, attention_mask)
            
            all_preds.append(accelerator.gather(pred))
            all_targets.append(accelerator.gather(target))
    
    all_preds   = torch.cat(all_preds)[:len(data_loader.dataset)].flatten()
    all_targets = torch.cat(all_targets)[:len(data_loader.dataset)]

    print(f"all_preds.shape {all_preds.shape} | all_targets.shape {all_targets.shape}")

    mse_sum = nn.MSELoss(reduction="sum")(all_preds, all_targets).item()
    return mse_sum / len(data_loader.dataset)

def predict(model, data_loader, device):
    """Returns an np.array with predictions of the |model| on |data_loader|"""
    model.eval()

    result = np.zeros(len(data_loader.dataset))    
    index = 0
    
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
                        
            pred = model(input_ids, attention_mask)                        

            result[index : index + pred.shape[0]] = pred.flatten().to("cpu")
            index += pred.shape[0]

    return result

def create_optimizer(model, learning_rate):
    named_parameters = list(model.named_parameters())    
    
    roberta_parameters = named_parameters[:197]    
    attention_parameters = named_parameters[199:203]
    regressor_parameters = named_parameters[203:]
        
    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})

    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = learning_rate

        if layer_num >= 69:        
            lr = lr * 2.5

        if layer_num >= 133:
            lr = lr * 5

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)