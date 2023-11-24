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

from .av_net import AVNet
from torch.nn.utils.rnn import pad_sequence
import copy

def setupavsr_model(tokenizer, train_config, model_config, **kwargs):
    return avsrllm_model(tokenizer, train_config, model_config, **kwargs)


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    if model_config.llm_name=="llama-2-7b-hf":
        tokenizer = LlamaTokenizer.from_pretrained(model_config.llm_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 2
        return tokenizer


def extract_variable_length_features(self, x: torch.Tensor):  #torch.Size([2, 80, 371])
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))  #torch.Size([2, 512, 371])
    x = F.gelu(self.conv2(x))  #torch.Size([2, 512, 186])
    x = x.permute(0, 2, 1)  #torch.Size([2, 186, 512])

    # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
    # x = (x + self.positional_embedding).to(x.dtype)
    x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

    for block in self.blocks:
        x = block(x)

    x = self.ln_post(x)
    return x #torch.Size([2, 186, 512])

def setup_llm(train_config, model_config, **kwargs):
    from pkg_resources import packaging
    use_cache = False if train_config.enable_fsdp else None  #None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp: 
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(model_config.llm_path)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:  #
        model = LlamaForCausalLM.from_pretrained(
            model_config.llm_path,
            load_in_8bit=True if train_config.quantization else None,  #train_config.quantization: true
            device_map="auto" if train_config.quantization else None, 
            use_cache=use_cache,
        )
    if train_config.enable_fsdp and train_config.use_fast_kernels:  #x
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:  # 
        model = prepare_model_for_kbit_training(model)

    if train_config.use_peft:  #x
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


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
        
        # audio-visual 
        self.avnet=AVNet(model_config)
        
        # load_ckpt
        checkpoint = torch.load(model_config.TRAIN_LRS3_MODEL_FILE)
        self.avnet.load_state_dict(checkpoint['state_dict'],strict=False)    # 最终输出ctc/attention的模块没有用到

        # if not model_config.modal == "AV" and model_config.TRAIN_LRS3_MODEL_FILE is not None:
        #     stateDict = torch.load(model_config.TRAIN_LRS3_MODEL_FILE, map_location="cpu")['state_dict']  #When you call torch.load() on a file which contains GPU tensors, those tensors will be loaded to GPU by default. You can call torch.load(.., map_location='cpu') and then load_state_dict() to avoid GPU RAM surge when loading a model checkpoint.
        #     model.load_state_dict(stateDict, strict=False)

        # if model_config.modal == "AV" and model_config.TRAINED_AO_FILE is not None and model_config.TRAINED_VO_FILE is not None:
        #     AOstateDict = torch.load(model_config.TRAINED_AO_FILE)['state_dict']
        #     stateDict = torch.load(model_config.TRAINED_VO_FILE)['state_dict']
        #     for k in list(AOstateDict.keys()):
        #         if not (k.startswith('audioConv') or k.startswith('wav2vecModel')):
        #             del AOstateDict[k]

        #     for k in list(stateDict.keys()):
        #         if not (k.startswith('videoConv') or k.startswith('visualModel')):
        #             del stateDict[k]
        #     stateDict.update(AOstateDict)
        #     model.load_state_dict(stateDict, strict=False)

        # 直接load一个完整的 做初始化


        # freeze
        for name, param in self.avnet.named_parameters(): 
            param.requires_grad = False       
        self.avnet.eval()


        # llama
        self.llm = setup_llm(train_config, model_config, **kwargs)

        # projector
        self.feature_projector = nn.Linear(model_config.DMODEL, self.llm.config.hidden_size)  #(1024,4096)


        # # tokenizer
        # self.tokenizer = tokenizer
    
    def forward(self, inputBatch0, inputBatch1,inputBatch2,inputBatch3,  targetoutBatch, targetLenBatch, maskw2v, **kwargs,):
    #def forward(self, inputBatch,targetinBatch, targetLenBatch, maskw2v, **kwargs,):
        inputBatch=(inputBatch0, inputBatch1,inputBatch2,inputBatch3)   # targetinBatch是前面加

        jointBatch, inputLenBatch, mask = self.avnet(inputBatch, maskw2v)  #[129, 2, 1024], [129,125], [2,129] mask false的地方是不mask的，mask的位置是true , 就mask[1]末尾4个true  #输出应该是 bs,l,dim
        jointBatch = jointBatch.transpose(0, 1)  #(2,129,1024)
            
        # project
        feature_tokens = self.feature_projector(jointBatch)  #(2,129,4096)

        if hasattr(self.llm.model, "embed_tokens"):
            texts_embeds = self.llm.model.embed_tokens(targetoutBatch)
        else: #
            texts_embeds = self.llm.model.model.embed_tokens(targetoutBatch)  #(2,37)-> (2,37,4096)

        #还原原来长度
        #搞出每个item的特征和文本 拼起来 再padding

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


        # labels_list=[]
        # for i in range(jointBatch.size(0)): 
        #     labels= torch.cat(( torch.full((inputLenBatch[i],),self.IGNORE_INDEX , device=targetoutBatch.device) ,  targetoutBatch[i][:targetLenBatch[i]]) ,dim=0)   
        #     labels_list.append((labels))
        labels_list= [  torch.cat(( torch.full((inputLenBatch[i],),self.IGNORE_INDEX , device=targetoutBatch.device) ,  targetoutBatch[i][:targetLenBatch[i]]) ,dim=0)    for i in range(jointBatch.size(0))     ]  #[166,151]
        labels = pad_sequence(labels_list, batch_first=True, padding_value=self.IGNORE_INDEX)  #(2,166)


        # 研究 attention_mask，labels,里面实现了错位
        # 
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask = mask2, labels=labels)  #self PeftModelForCausalLM

        return model_outputs  #logits:[2,292,32000]  #loss:6.9475

