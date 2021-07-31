import os, time, math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (-1., 4)]

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True

def create_folds(data: pd.DataFrame, num_splits: int, seed: int):
    data["kfold"] = -1
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)
    kf = StratifiedKFold(n_splits=num_splits)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    data = data.drop("bins", axis=1)
    return data


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

def create_optimizer_roberta_large(model, learning_rate):
    """
    Code is copied from https://www.kaggle.com/jcesquiveld/roberta-large-5-fold-single-model-meanpooling/data#1386802
    """
    lr = learning_rate
    multiplier = 0.975
    classifier_lr = 2e-5 # copied from the same notebook comment

    parameters = []
    for layer in range(23,-1,-1):
        layer_params = {
            'params': [p for n,p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier
    classifier_params = {
        'params': [p for n,p in model.named_parameters() if 'layer_norm' in n or 'linear' in n 
                   or 'pooling' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)
    return AdamW(parameters)


def train(model, model_path, train_loader, val_loader,
          optimizer, device, scheduler=None, num_epochs=3, standard_error_alpha=None):    
    best_val_rmse = None
    best_epoch = 0
    step = 0
    last_eval_step = 0
    eval_period = EVAL_SCHEDULE[0][1]    

    start = time.time()

    for epoch in range(num_epochs):                           
        val_rmse = None         

        for batch_num, (input_ids, attention_mask, target, standard_error) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)            
            target = target.to(device)
            standard_error = standard_error.to(device)
            optimizer.zero_grad()
            model.train()

            if standard_error_alpha:
                pred, pred_se = model(input_ids, attention_mask)
                mse = nn.MSELoss(reduction="mean")(pred.flatten(), target)
                mse_se = nn.MSELoss(reduction="mean")(pred_se.flatten(), standard_error)
                mse = (1. - standard_error_alpha) * mse + standard_error_alpha * mse_se
            else:
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
                
                val_rmse = math.sqrt(eval_mse(model, val_loader, device, standard_error_alpha))                            

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


def eval_mse(model, data_loader, device, standard_error_alpha=None):
    """Evaluates the mean squared error of the |model| on |data_loader|"""
    model.eval()            
    mse_sum = 0

    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, target, standard_error) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)                        
            target = target.to(device)           
            
            if standard_error_alpha:
                pred, pred_se = model(input_ids, attention_mask)
            else:
                pred = model(input_ids, attention_mask)

            mse_sum += nn.MSELoss(reduction="sum")(pred.flatten(), target).item()
                

    return mse_sum / len(data_loader.dataset)

def predict(model, data_loader, device, standard_error_alpha=None):
    """Returns an np.array with predictions of the |model| on |data_loader|"""
    model.eval()

    result = np.zeros(len(data_loader.dataset))    
    index = 0
    
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
                        
            if standard_error_alpha:
                pred, pred_se = model(input_ids, attention_mask)
            else:
                pred = model(input_ids, attention_mask)

            result[index : index + pred.shape[0]] = pred.flatten().to("cpu")
            index += pred.shape[0]

    return result

def rmse(ground_truth, prediction):
    return np.sqrt(mean_squared_error(ground_truth, prediction))