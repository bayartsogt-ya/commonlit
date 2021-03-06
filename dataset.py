import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class LitDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, inference_only=False, mask_prob=.0):
        super().__init__()

        self.df = df        
        self.inference_only = inference_only
        self.text = df.excerpt.tolist()
        self.mask_prob = mask_prob
        self.max_len = max_len
        #self.text = [text.replace("\n", " ") for text in self.text]
        
        if not self.inference_only:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)
            self.standard_error = torch.tensor(df.standard_error.values, dtype=torch.float32)

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = max_len,
            truncation = True,
            return_attention_mask=True
        )

        if mask_prob > .0:
            print("Using attention mask:", mask_prob)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(
            self.encoded['attention_mask'][index] * (np.random.random((self.max_len,)) > self.mask_prob))
        
        if self.inference_only:
            return (input_ids, attention_mask)            
        else:
            target = self.target[index]
            se = self.standard_error[index]
            return (input_ids, attention_mask, target, se)