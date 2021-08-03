# CommonLit Readability Prize (🥈 63/3682)
[![Screen Shot 2021-08-03 at 11 37 30 AM](https://user-images.githubusercontent.com/43239645/128053195-dc626c44-612f-4e9e-aac4-498b77c86033.png)](https://www.kaggle.com/c/commonlitreadabilityprize)

## What you can find useful
- Transformers fine-tuning from modelhub (thanks to [huggingface](https://hf.co))
- Huggingface_hub integration (thanks to [huggingface_hub](https://github.com/huggingface/huggingface_hub))
- More mask on Attention Mask

## Results
| Model | CV | LB |
| - | - | - |
| GPT-2 Medium | 0.502 | - |
| RoBerta-base | 0.479 | - |
| RoBerta-large | 0.479 | 0.473 |
| Electra Large | 0.480 | 0.469 |
| FBMuppet RoBerta Large | 0.486 | 0.480 |
| **Ensemble** | **0.457**  | **0.455** |

## Train
If you want to use `push-to-hub` then you need to login to huggingface:
```
huggingface-cli login
```

Then start training on only fold 0. After fold, automatic evaluation takes a place.
```
python main.py \
    --fold 0 \
    --model-path roberta-large \
    --lr-scheduler linear \
    --model-type attention_head \
    --warmup-steps 0 \
    --num-epochs 5 \
    --batch-size 8 \
    --learning-rate 3e-5 \
    --seed 1000 \
    --roberta-large-optimizer \
    --mask-prob 0.1 \
    --do-train \
    --push-to-hub
```

## Ideas that did not work
- Backtranslation & Pseudo-labeling
- Smaller MAX_LEN and Bigger Batch size
- Bigger mask-prob

## References:
Special thanks to great resources:
- https://www.kaggle.com/c/commonlitreadabilityprize
- https://hf.co
- https://github.com/huggingface/huggingface_hub
- https://github.com/Kaggle/kaggle-api
- https://www.kaggle.com/abhishek/step-1-create-folds
- https://www.kaggle.com/andretugan/lightweight-roberta-solution-in-pytorch
- https://www.kaggle.com/jcesquiveld/roberta-large-5-fold-single-model-meanpooling
