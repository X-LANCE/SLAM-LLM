#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    RobertaModel,
    RobertaTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    CLIPTokenizer,
    CLIPTextModel,
)

MODELS = {
    'openai/clip-vit-base-patch32': (CLIPTextModel, CLIPTokenizer, 512),
    'prajjwal1/bert-tiny': (BertModel, BertTokenizer, 128),
    'prajjwal1/bert-mini': (BertModel, BertTokenizer, 256),
    'prajjwal1/bert-small': (BertModel, BertTokenizer, 512),
    'prajjwal1/bert-medium': (BertModel, BertTokenizer, 512),
    'gpt2': (GPT2Model, GPT2Tokenizer, 768),
    'distilgpt2': (GPT2Model, GPT2Tokenizer, 768),
    'bert-base-uncased': (BertModel, BertTokenizer, 768),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024),
    'roberta-base': (RobertaModel, RobertaTokenizer, 768),
    'roberta-large': (RobertaModel, RobertaTokenizer, 1024),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768),
    "distilroberta-base": (RobertaModel, RobertaTokenizer, 768),
}


class TextEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.tokenizer = MODELS[config["text_encoder_args"]["type"]][1].from_pretrained(
            config["text_encoder_args"]["type"])
        self.text_encoder = MODELS[config["text_encoder_args"]["type"]][0].from_pretrained(
            config["text_encoder_args"]["type"],
            add_pooling_layer=False)

        if config["text_encoder_args"]["freeze"]:
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad = False

        self.text_width = MODELS[config["text_encoder_args"]["type"]][-1]

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, text):

        text_input = self.tokenizer(text,
                                    padding='longest',
                                    truncation=True,
                                    max_length=30,
                                    return_tensors="pt").to(self.device)
        text_output = self.text_encoder(input_ids=text_input.input_ids,
                                        attention_mask=text_input.attention_mask)[0]
        return text_output, text_input.attention_mask
