import types
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
import whisper

from llama_recipes.utils.config_utils import generate_peft_config
from llama_recipes.utils.train_utils import print_model_size

from .AV.av_net import AVNet
from .slam_model import setup_llm
from torch.nn.utils.rnn import pad_sequence
import copy
from llama_recipes.utils.metric import compute_accuracy

def setupavsr_model(tokenizer, train_config, model_config, **kwargs):
    return avsrllm_model(tokenizer, train_config, model_config, **kwargs)

class avsrllm_model(nn.Module):
    def __init__(
        self,
        tokenizer, 
        train_config, 
        model_config, 
        **kwargs
    ):
        super().__init__()

        self.IGNORE_INDEX = -100 # The default setting in CrossEntropyLoss
         
        # audio-visual   ↓
        self.avnet=AVNet(model_config)
        
        # load_ckpt ↑
        checkpoint = torch.load(model_config.TRAIN_LRS3_MODEL_FILE)
        self.avnet.load_state_dict(checkpoint['state_dict'],strict=False)    # 最终输出ctc/attention的模块没有用到

        # freeze 外面都有
        for name, param in self.avnet.named_parameters(): 
            param.requires_grad = False       
        self.avnet.eval()

        # llama
        self.llm = setup_llm(train_config, model_config, **kwargs)

        # projector
        self.feature_projector = nn.Linear(model_config.DMODEL, self.llm.config.hidden_size)  #(512,4096)  好像有遗留问题 TO DO

        # tokenizer
        self.tokenizer = tokenizer   #tokenizer = LlamaTokenizer.from_pretrained(model_config.llm_path) 不需要保存
        self.metric = kwargs.get("metric", "acc")
    
    def forward(self, inputBatch0,inputBatch1,inputBatch2,inputBatch3,  targetoutBatch, targetLenBatch, maskw2v, **kwargs):
        inputBatch=(inputBatch0, inputBatch1,inputBatch2,inputBatch3)   # targetinBatch是前面加

        jointBatch, inputLenBatch, mask = self.avnet(inputBatch, maskw2v)  #[129, 2, 1024], [129,125], [2,129] mask false的地方是不mask的，mask的位置是true , 就mask[1]末尾4个true  #输出应该是 bs,l,dim
        jointBatch = jointBatch.transpose(0, 1)  #(2,129,1024)
            
        # project
        feature_tokens = self.feature_projector(jointBatch)  #(2,129,4096)

        if hasattr(self.llm.model, "embed_tokens"):
            texts_embeds = self.llm.model.embed_tokens(targetoutBatch)
        else: #
            texts_embeds = self.llm.model.model.embed_tokens(targetoutBatch)  #(2,37)-> (2,37,4096)

        #还原原来长度 搞出每个item的特征和文本 拼起来 再padding

        #input_list=[torch.cat( (jointBatch[i, ~mask[i]] , targetoutBatch[i][:targetLenBatch[i]]),  dim=1) for i in range(jointBatch.size(0) )]
        # for i in range(jointBatch.size(0)): 
        #     a= feature_tokens[i, ~mask[i]]  #(129,4096) (125,4096)
        #     b= texts_embeds[i][:targetLenBatch[i]][:] #(37,4096) (26,4096)
        #     input= torch.cat( (a,b),  dim=0)  #(166,4096) (151,4096)

        input_lists=[torch.cat(  (feature_tokens[i, ~mask[i]], texts_embeds[i][:targetLenBatch[i]][:] ) , dim=0  ) for i in range(jointBatch.size(0)) ]
        inputs_embeds = pad_sequence(input_lists, batch_first=True, padding_value=0)  #(2,166,4096)

        lengths=[item.size(0) for item in input_lists]  #[166, 151]
        max_length=max(lengths)  #166
        mask2 = torch.zeros(len(input_lists),max_length,dtype=torch.bool)  #(2,166)
        for i,length in enumerate(lengths):
            mask2[i,:length]=1  #mask的地方是false，其余是true，只有maks2[1]末尾有15个false   
        mask2=mask2.to("cuda:0")


        # labels_list=[]
        # for i in range(jointBatch.size(0)): 
        #     labels= torch.cat(( torch.full((inputLenBatch[i],),self.IGNORE_INDEX , device=targetoutBatch.device) ,  targetoutBatch[i][:targetLenBatch[i]]) ,dim=0)   
        #     labels_list.append((labels))
        labels_list= [  torch.cat(( torch.full((inputLenBatch[i],),self.IGNORE_INDEX , device=targetoutBatch.device) ,  targetoutBatch[i][:targetLenBatch[i]]) ,dim=0)    for i in range(jointBatch.size(0))     ]  #[166,151]
        labels = pad_sequence(labels_list, batch_first=True, padding_value=self.IGNORE_INDEX)  #(2,166)


        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask = mask2, labels=labels)  #self PeftModelForCausalLM 里面实现了错位

        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)

        return model_outputs, acc  #logits:[2,292,32000]  #loss:6.9475

    def save_pretrained(self, output_dir):
        save_dir= output_dir+'/avsrmodel.pt'
        self.llm.save_pretrained(output_dir)
        modules_to_save={
            'avnet': self.avnet.state_dict(),
            'feature_projector':self.feature_projector.state_dict(),
        }

        torch.save(modules_to_save,save_dir)


