from transformers import Blip2QFormerConfig, Blip2QFormerModel
import torch.nn as nn
import torch
from .utils import positionalencoding1d

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config, use_pe=False):
        super().__init__()
        self.encoder_dim = config['encoder_dim']
        self.output_dim = config['output_dim']
        self.query_len = config['query_len']
        self.use_pe = use_pe
        if use_pe: 
            self.pe = positionalencoding1d(self.query_len, self.encoder_dim)

        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = 2

        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim, eps=1e-5)

    def forward(self, x, atts):
        if self.use_pe: 
            query = query + self.pe
            
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj