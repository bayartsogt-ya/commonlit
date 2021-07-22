import os
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold

import gc

# local imports 
from utils import set_random_seed, eval_mse, predict, create_optimizer
from dataset import LitDataset
from model import LitModel

gc.enable()

NUM_FOLDS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LEN = 248
NUM_DATA_WORKERS = 2
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]
# ROBERTA_PATH = "../input/clrp-roberta-base/clrp_roberta_base"
ROBERTA_PATH = "roberta-base"
OUTPUT_DIR = "."
KAGGLE_DATASET_ID = "cl-roberta-base-v1"
KAGGLE_DATASET_TITLE = "cl-roberta-base-v1"

DATA_DIR = "data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, model_path, train_loader, val_loader,
          optimizer, scheduler=None, num_epochs=NUM_EPOCHS):    
    best_val_rmse = None
    best_epoch = 0
    step = 0
    last_eval_step = 0
    eval_period = EVAL_SCHEDULE[0][1]    

    start = time.time()

    for epoch in range(num_epochs):                           
        val_rmse = None         

        for batch_num, (input_ids, attention_mask, target) in enumerate(train_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)            
            target = target.to(DEVICE)
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
                
                val_rmse = math.sqrt(eval_mse(model, val_loader, DEVICE))                            

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

if __name__ == "__main__":
    
    # ----------------------------- DATA --------------------------------
    print("loading data")
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    # Remove incomplete entries if any.
    train_df.drop(train_df[(train_df.target == 0) & (train_df.standard_error == 0)].index,
                inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    submission_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    print("data loaded!")

    # ----------------------------- TOKENIZER --------------------------------
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"TOKENIZERS_PARALLELISM = {os.environ['TOKENIZERS_PARALLELISM']}")

    AutoConfig.from_pretrained("roberta-base").save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    gc.collect()

    # ----------------------------- 5-FOLD TRAINING --------------------------------
    print("Starting training...")
    SEED = 1000
    list_val_rmse = []

    kfold = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_df)):    
        print(f"\nFold {fold + 1}/{NUM_FOLDS}")
        model_path = f"{OUTPUT_DIR}/model_{fold + 1}.pth"
            
        set_random_seed(SEED + fold)
        
        train_dataset = LitDataset(train_df.loc[train_indices], tokenizer, MAX_LEN)    
        val_dataset = LitDataset(train_df.loc[val_indices], tokenizer, MAX_LEN)    
            
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                drop_last=True, shuffle=True, num_workers=NUM_DATA_WORKERS)    
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                drop_last=False, shuffle=False, num_workers=NUM_DATA_WORKERS)    
            
        set_random_seed(SEED + fold)    

        model = LitModel(ROBERTA_PATH).to(DEVICE)

        optimizer = create_optimizer(model, LEARNING_RATE)                        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=NUM_EPOCHS * len(train_loader),
            num_warmup_steps=50)    

        list_val_rmse.append(train(model, model_path, train_loader,
                                val_loader, optimizer, scheduler=scheduler))

        del model
        gc.collect()

        print("\nPerformance estimates:")
        print(list_val_rmse)
        print("Mean:", np.array(list_val_rmse).mean())
        
    with open(f"{OUTPUT_DIR}/log.txt") as writer:
        print("\nPerformance estimates:")
        print(list_val_rmse)
        print("Mean:", np.array(list_val_rmse).mean())
        writer.write("\nPerformance estimates:")
        writer.write("\n")
        writer.write(list_val_rmse)
        writer.write("\n")
        writer.write("Mean:", np.array(list_val_rmse).mean())


    # ----------------------------- KAGGLE DATASETS -----------------------
    print("Uploading to Kaggle...")
    import json
    import subprocess
    kaggle_config = {
        "licenses": [
            {
            "name": "CC0-1.0"
            }
        ], 
        "id": f"bayartsogtya/{KAGGLE_DATASET_ID}", 
        "title": KAGGLE_DATASET_TITLE
    }
    with open(f"{OUTPUT_DIR}/dataset-metadata.json", "w") as writer:
        json.dump(kaggle_config, writer, indent=4)

    subprocess.run(["kaggle", "datasets", "create", "-p", OUTPUT_DIR])