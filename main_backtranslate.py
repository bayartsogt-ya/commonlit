import os, gc, time, argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold

from huggingface_hub.hf_api import HfFolder, HfApi
from huggingface_hub.repository import Repository

# local imports 
from utils import set_random_seed, eval_mse, predict, create_optimizer, train, create_folds, create_optimizer_roberta_large
from dataset import LitDataset
from model import AttentionHeadModel, MeanPoolingModel

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
        --back-translate \
        --use-warmup-scheduler
    """
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--fold", type=int, default=0, help="If passed, will train only this fold.")

    # model selection
    parser.add_argument("--model-path", type=str, default="roberta-base", help="Hugginface base model name")
    parser.add_argument("--lr-scheduler", type=str, default="cosine", help="If passed, Warm Up scheduler will be used")
    parser.add_argument("--model-type", type=str, default="attention_head", help="One of [attention_head, mean_pooling]")

    # hyperparameter
    parser.add_argument("--num-epochs", type=int, default=3, help="If passed, Number of Epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="If passed, seed will be used for reproducability")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="If passed, seed will be used for reproducability")
    parser.add_argument("--seed", type=int, default=1000, help="If passed, seed will be used for reproducability")
    parser.add_argument("--back-translate", action="store_true", default=False, help="If passed, Back translated data will be added")
    parser.add_argument("--warmup-steps", type=int, default=40, help="If passed, Warm Up scheduler will be used")
    parser.add_argument("--roberta-large-optimizer", action="store_true", default=False, help="If passed, ")
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
    OUTPUT_DIR = f"kaggle-{MODEL_PATH.replace('/','-')}-" + \
                    f"{'ah' if args.model_type == 'attention_head' else 'mp'}-" + \
                    f"{'bt' if args.back_translate else 'orig'}-"+ \
                    f"s{SEED}"


    # ----------------------------- HF API --------------------------------
    hf_token = HfFolder.get_token(); api = HfApi()
    repo_link = api.create_repo(token=hf_token, name=OUTPUT_DIR, exist_ok=True, private=True)
    repo = Repository(local_dir=OUTPUT_DIR, clone_from=repo_link, use_auth_token=hf_token, git_user=GIT_USER, git_email=GIT_EMAIL)
    print("[success] configured HF Hub to", OUTPUT_DIR)

    with open(f"{OUTPUT_DIR}/training_arguments.json", "w") as writer:
        json.dump(vars(args), writer, indent=4)
        print(f"[success] wrote training args to {OUTPUT_DIR}")

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

    # ---------------- MODEL SELECTION ----------------
    if args.model_type == "attention_head":
        model = AttentionHeadModel(MODEL_PATH).to(DEVICE)
    elif args.model_type == "mean_pooling":
        model = MeanPoolingModel(MODEL_PATH).to(DEVICE)
    else:
        raise Exception("`model-type` should be one of [attention_head, mean_pooling]")
    
    # ---------------- OPTIMIZER SELECTION ----------------
    if args.roberta_large_optimizer:
        print("Using Roberta Large Optimizer copied from https://www.kaggle.com/jcesquiveld/roberta-large-5-fold-single-model-meanpooling/notebook")
        optimizer = create_optimizer_roberta_large(model, LEARNING_RATE)
    else:
        optimizer = create_optimizer(model, LEARNING_RATE)

    
    # ---------------- SCHEDULER SELECTION ----------------
    print(f"Using {args.lr_scheduler} scheduler")
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=NUM_EPOCHS * len(train_loader),
            num_warmup_steps=args.warmup_steps)
    elif args.lr_scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=NUM_EPOCHS * len(train_loader),
            num_warmup_steps=args.warmup_steps)
    else:
        scheduler = None

    # ---------------- TRAINING ----------------
    val_rmse = train(model, model_output_path, train_loader,
                            val_loader, optimizer, DEVICE, scheduler=scheduler, num_epochs=NUM_EPOCHS)

    del model
    gc.collect()

    print("\nPerformance estimates:")
    print(f"FOLD {FOLD} | VAL RMSE: {val_rmse}")

    with open(f"{OUTPUT_DIR}/log_{FOLD}.txt", "w") as writer:
        writer.write(f"""
        Performance estimates:
        FOLD {FOLD} | VAL RMSE: {val_rmse}
        """)

    # ----------------------------- UPLOAD TO HUB -----------------------
    repo.git_pull() # get updates first
    commit_link = repo.push_to_hub(commit_message=f"MODEL={FOLD} VALID={val_rmse:.3f}") # then push
    print("[success] UPLOADED TO HUGGINGFACE HUB", commit_link)
    print("[success] TIME SPENT: %.3f" % (time.time()-start_time))