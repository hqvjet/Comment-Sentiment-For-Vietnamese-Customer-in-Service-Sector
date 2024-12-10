import torch
import torch.nn as nn
import json

with open('models/Transformer/config.json', 'r') as file:
    config = json.load(file)

phobert_config = config['phobert']
phow2v_config = config['phow2v']
num_classes = 3

class Transformer(nn.Module):
    def __init__(self, device, input_shape, emb_tech, dropout=0.1):
        super(Transformer, self).__init__()
        config = phobert_config if emb_tech == 1 else phow2v_config
        self.model_name = 'Transformer'
        self.hidden_size = input_shape[-1]
        self.num_layers = config['num_layers']
        self.emb_tech = emb_tech

        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=6, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.emb_tech == 1:
            x = x.unsqueeze(1)

        attn_output, _ = self.attention(x, x, x)
        pooled = attn_output.mean(dim=1)

        out = self.fc(pooled)
        out = self.softmax(out)

        return out
