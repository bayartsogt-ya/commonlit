import os, time, math
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW

EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True


def eval_mse(model, data_loader, device):
    """Evaluates the mean squared error of the |model| on |data_loader|"""
    model.eval()            
    mse_sum = 0

    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, target) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)                        
            target = target.to(device)           
            
            pred = model(input_ids, attention_mask)                       

            mse_sum += nn.MSELoss(reduction="sum")(pred.flatten(), target).item()
                

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



def train(model, model_path, train_loader, val_loader,
          optimizer, device, scheduler=None, num_epochs=3):    
    best_val_rmse = None
    best_epoch = 0
    step = 0
    last_eval_step = 0
    eval_period = EVAL_SCHEDULE[0][1]    

    start = time.time()

    for epoch in range(num_epochs):                           
        val_rmse = None         

        for batch_num, (input_ids, attention_mask, target) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)            
            target = target.to(device)
            optimizer.zero_grad()
            model.train()
            pred = model(input_ids, attention_mask)
            mse = nn.MSELoss(reduction="mean")(pred.flatten(), target)
                        
            mse.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()
            
            if step >= last_eval_step + eval_period:
                # Evaluate the model on val_loader.
                elapsed_seconds = time.time() - start
                num_steps = step - last_eval_step
                print(f"\n{num_steps} steps took {elapsed_seconds:0.3} seconds")
                last_eval_step = step
                
                val_rmse = math.sqrt(eval_mse(model, val_loader, device))                            

                print(f"Epoch: {epoch} batch_num: {batch_num}", 
                      f"val_rmse: {val_rmse:0.4}")

                for rmse, period in EVAL_SCHEDULE:
                    if val_rmse >= rmse:
                        eval_period = period
                        break                               
                
                if not best_val_rmse or val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_path)
                    print(f"New best_val_rmse: {best_val_rmse:0.4}")
                else:
                    print(f"Still best_val_rmse: {best_val_rmse:0.4}",
                          f"(from epoch {best_epoch})")
                start = time.time()
            step += 1
                        
    return best_val_rmse
