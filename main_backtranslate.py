import os, gc, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold

from huggingface_hub.hf_api import HfFolder, HfApi
from huggingface_hub.repository import Repository

# local imports 
from utils import set_random_seed, eval_mse, predict, create_optimizer, train, create_folds
from dataset import LitDataset
from model import LitModel

gc.enable()

OUTPUT_DIR = "kaggle-commonlit-exp-no0"
FOLD = 0
NUM_FOLDS = 5
SEED = 1000
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LEN = 248
NUM_DATA_WORKERS = 2
MODEL_PATH = "roberta-base"
DATA_DIR = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GIT_USER = "bayartsogt"
GIT_EMAIL = "bayartsogtyadamsuren@icloud.com"

if __name__ == "__main__":
    """
    python main_backtranslate.py \
        --fold 0 \
        --model-path roberta-base \
        --num-epochs 3 \
        --batch-size 16 \
        --learning-rate 2e-5 \
        --seed 1000 \
        --back-translate False
    """
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--fold", type=int, default=0, help="If passed, will train only this fold.")
    parser.add_argument("--model-path", type=str, default="roberta-base", help="Hugginface model name")
    parser.add_argument("--num-epochs", type=int, default=3, help="If passed, Number of Epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="If passed, seed will be used for reproducability")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="If passed, seed will be used for reproducability")
    parser.add_argument("--seed", type=int, default=1000, help="If passed, seed will be used for reproducability")
    parser.add_argument("--back-translate", type=bool, default=False, help="If passed, Back translated data will be added")
    parser.add_argument("--use-warmup-scheduler", type=bool, default=True, help="If passed, Back translated data will be added")
    args = parser.parse_args()

    print("----------- ARGS -----------")
    print(args)
    print("----------------------------")

    FOLD = args.fold
    NUM_EPOCHS = args.num_epochs
    MODEL_PATH = args.model_path
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    # for google/bigbird-roberta-base => google-bigbird-roberta-base
    OUTPUT_DIR = f"kaggle-{MODEL_PATH.replace('/','-')}-{'backtranlated' if args.back_translate else 'original'}-seed-{SEED}"

    # ----------------------------- HF API --------------------------------
    hf_token = HfFolder.get_token(); api = HfApi()
    repo_link = api.create_repo(token=hf_token, name=OUTPUT_DIR, exist_ok=True, private=True)
    repo = Repository(local_dir=OUTPUT_DIR, clone_from=repo_link, use_auth_token=hf_token, git_user=GIT_USER, git_email=GIT_EMAIL)
    print("[success] configured HF Hub to", OUTPUT_DIR)

    # ----------------------------- DATA --------------------------------
    print("loading data")
    if args.back_translate:
        languages = ["es","fr","gr","ja","mn"]
        train_df = pd.read_csv(f"{DATA_DIR}/train_backtranslated.csv")
        print("[Backtranslated] Train data shape:", train_df.shape)
    else:
        train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
        # Remove incomplete entries if any.
        train_df.drop(train_df[(train_df.target == 0) & (train_df.standard_error == 0)].index,
                    inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        print("Train data shape:", train_df.shape)

    submission_df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

    train_df = create_folds(train_df, NUM_FOLDS, SEED)

    print("[success] data loaded!")
    # ----------------------------- TOKENIZER --------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"TOKENIZERS_PARALLELISM = {os.environ['TOKENIZERS_PARALLELISM']}")

    if FOLD is not None and FOLD == 0:
        AutoConfig.from_pretrained(MODEL_PATH).save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    gc.collect()

    # ----------------------------- 5-FOLD TRAINING --------------------------------
    print("Starting training...")
    list_val_rmse = []
    # for fold, (train_indices, val_indices) in enumerate(kfold.split(train_df)):
    # if FOLD is not None and FOLD != fold:
    #     continue
    print(f"\nFold {FOLD}/{NUM_FOLDS}")
    model_output_path = f"{OUTPUT_DIR}/model_{FOLD}.pth"

    set_random_seed(SEED)

    _train_df = train_df.query("kfold!=@FOLD")
    _valid_df = train_df.query("kfold==@FOLD")

    if args.back_translate:
        _train_df_merged = _train_df[["id", "excerpt", "target"]].copy()
        for lan in languages:
            tmp = pd.DataFrame()
            tmp["id"] = _train_df["id"].copy()
            tmp["excerpt"] = _train_df[f"excerpt_{lan}"].copy()
            tmp["target"]  = _train_df[f"pred_{lan}"].copy()
            _train_df_merged = _train_df_merged.append(tmp)
        assert _train_df.shape[0] * 1. ==  _train_df_merged.shape[0] / (len(languages) + 1)
        _train_df = _train_df_merged

    train_dataset = LitDataset(_train_df, tokenizer, MAX_LEN)
    val_dataset = LitDataset(_valid_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            drop_last=True, shuffle=True, num_workers=NUM_DATA_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            drop_last=False, shuffle=False, num_workers=NUM_DATA_WORKERS)

    set_random_seed(SEED)    

    model = LitModel(MODEL_PATH).to(DEVICE)

    optimizer = create_optimizer(model, LEARNING_RATE)

    scheduler = None
    if args.use_warmup_scheduler:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=NUM_EPOCHS * len(train_loader),
            num_warmup_steps=50)

    list_val_rmse.append(train(model, model_output_path, train_loader,
                            val_loader, optimizer, DEVICE, scheduler=scheduler, num_epochs=NUM_EPOCHS))

    del model
    gc.collect()

    print("\nPerformance estimates:")
    print(list_val_rmse)
    print("Mean:", np.array(list_val_rmse).mean())

    with open(f"{OUTPUT_DIR}/log_{FOLD}.txt", "w") as writer:
        writer.write(f"""
        Performance estimates:
        {list_val_rmse}
        Mean: {np.array(list_val_rmse).mean()}
        """)

    # ----------------------------- UPLOAD TO HUB -----------------------
    repo.git_pull() # get updates first
    commit_link = repo.push_to_hub(commit_message=f"MODEL={FOLD} VALID={np.array(list_val_rmse).mean():.3f}") # then push
    print("[success] UPLOADED TO HUGGINGFACE HUB", commit_link)
    print("[success] TIME SPENT: %.3f".format(time.time()-start_time))