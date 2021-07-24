import os, gc, math, time, json, subprocess, argparse
import numpy as np
import pandas as pd
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold

from huggingface_hub.hf_api import HfFolder, HfApi
from huggingface_hub.repository import Repository

# local imports 
from utils import set_random_seed, eval_mse, predict, create_optimizer, train
from dataset import LitDataset
from model import LitModel

gc.enable()

OUTPUT_DIR = "kaggle-commonlit-exp-no0"
FOLD = 0
NUM_FOLDS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LEN = 248
NUM_DATA_WORKERS = 2
ROBERTA_PATH = "roberta-base"
DATA_DIR = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GIT_USER = "bayartsogt"
GIT_EMAIL = "bayartsogtyadamsuren@icloud.com"

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--fold", type=int, default=0, help="If passed, will train only this fold.")
    parser.add_argument("--output-dir", type=str, default="kaggle-commonlit-exp-no0", help="If passed, All the files will be saved to here and later pushed to HF Hub")
    parser.add_argument("--num-epochs", type=int, default=3, help="If passed, Number of Epochs to train")
    args = parser.parse_args()

    FOLD = args.fold
    OUTPUT_DIR = args.output_dir
    NUM_EPOCHS = args.num_epochs

    # ----------------------------- HF API --------------------------------
    hf_token = HfFolder.get_token(); api = HfApi()
    repo_link = api.create_repo(token=hf_token, name=OUTPUT_DIR, exist_ok=True, private=True)
    repo = Repository(local_dir=OUTPUT_DIR, clone_from=repo_link, use_auth_token=hf_token, git_user=GIT_USER, git_email=GIT_EMAIL)

    # ----------------------------- DATA --------------------------------
    print("loading data")
    # train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    # # Remove incomplete entries if any.
    # train_df.drop(train_df[(train_df.target == 0) & (train_df.standard_error == 0)].index,
    #             inplace=True)
    # train_df.reset_index(drop=True, inplace=True)
    # test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    submission_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

    languages = ["es","fr","gr","ja","mn"]
    train_df = pd.read_csv(f"{DATA_DIR}/train_backtranslated.csv")
    print("Backtranslated shape:", train_df.shape)

    print("data loaded!")
    # ----------------------------- TOKENIZER --------------------------------
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"TOKENIZERS_PARALLELISM = {os.environ['TOKENIZERS_PARALLELISM']}")

    if FOLD is not None and FOLD == 0:
        AutoConfig.from_pretrained("roberta-base").save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    gc.collect()

    # ----------------------------- 5-FOLD TRAINING --------------------------------
    print("Starting training...")
    SEED = 1000
    list_val_rmse = []

    kfold = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_df)):
        if FOLD is not None and FOLD != fold:
            continue

        print(f"\nFold {fold + 1}/{NUM_FOLDS}")
        model_path = f"{OUTPUT_DIR}/model_{fold + 1}.pth"

        set_random_seed(SEED + fold)

        _train_df = train_df.loc[train_indices]
        _valid_df = train_df.loc[val_indices]
        _train_df_merged = _train_df[["id", "excerpt", "target"]].copy()

        for lan in languages:
            tmp = pd.DataFrame()
            tmp["id"] = _train_df["id"].copy()
            tmp["excerpt"] = _train_df[f"excerpt_{lan}"].copy()
            tmp["target"]  = _train_df[f"pred_{lan}"].copy()
            _train_df_merged = _train_df_merged.append(tmp)

        assert _train_df.shape[0] * 1. ==  _train_df_merged.shape[0] / (len(languages) + 1)

        train_dataset = LitDataset(_train_df_merged, tokenizer, MAX_LEN)
        val_dataset = LitDataset(_valid_df, tokenizer, MAX_LEN)

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
                                val_loader, optimizer, DEVICE, scheduler=scheduler, num_epochs=NUM_EPOCHS))

        del model
        gc.collect()

        print("\nPerformance estimates:")
        print(list_val_rmse)
        print("Mean:", np.array(list_val_rmse).mean())

    with open(f"{OUTPUT_DIR}/log_{FOLD if FOLD is not None else -1}.txt", "w") as writer:
        print("\nPerformance estimates:")
        print(list_val_rmse)
        print("Mean:", np.array(list_val_rmse).mean())
        writer.write("\nPerformance estimates:")
        writer.write("\n")
        writer.write(list_val_rmse)
        writer.write("\n")
        writer.write("Mean:", np.array(list_val_rmse).mean())

    # ----------------------------- UPLOAD TO HUB -----------------------
    commit_link = repo.push_to_hub()
    print("UPLOADED TO HUGGINGFACE HUB", commit_link)
    print("TIME SPENT: %.3f".format(time.time()-start_time))