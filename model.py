import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoConfig

class AttentionHeadModel(nn.Module):
    def __init__(self, roberta_path):
        super().__init__()

        config = AutoConfig.from_pretrained(roberta_path)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(roberta_path, config=config)
        
        self.attention = nn.Sequential(            
            nn.Linear(config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        

        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, 1)
        )
        

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)        

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)
                
        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768        
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)        
        
        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)


class AttentionHeadModelWithStandardError(nn.Module):
    def __init__(self, roberta_path):
        super().__init__()

        config = AutoConfig.from_pretrained(roberta_path)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(roberta_path, config=config)
        
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        

        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, 1)
        )

        self.regressor_se = nn.Sequential( # for standard_error
            nn.Linear(config.hidden_size, 1)
        )
        

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)        
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        return self.regressor(context_vector), self.regressor_se(context_vector)


class MeanPoolingModel(nn.Module):
    """
    Idea is copied from here https://www.kaggle.com/jcesquiveld/roberta-large-5-fold-single-model-meanpooling/notebook
    """
    def __init__(self, model_name):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        norm_mean_embeddings = self.layer_norm(mean_embeddings)
        logits = self.linear(norm_mean_embeddings)
        
        return logits

class MLPHeadModel(nn.Module):
    """
    Simple MLP Head
    """
    def __init__(self, model_name):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.linear = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(input_ids, attention_mask)
        pooler_output = outputs[-1]
        logits = self.linear(pooler_output)
        
        return logits