from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from transformers import WhisperModel,AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from typing import List, Optional
from slam_llm.utils.metric import compute_accuracy


class EncoderProjectorQFormer(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = 1280
        configuration.num_hidden_layers = 8

        self.query = nn.Parameter(torch.zeros(1, 80, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, 3584)
        self.norm = nn.LayerNorm(3584, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj

class CustomSLM(PreTrainedModel):
    def __init__(self, config, ckpt_path=None):
        super().__init__(config)
        # 例如：
        self.encoder = WhisperModel.from_pretrained("openai/whisper-large-v3").encoder
        self.llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
        self.encoder_projector = EncoderProjectorQFormer()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

        if ckpt_path is not None:
            print("loading model checkpoint from: {}".format(ckpt_path))
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(ckpt_dict, strict=False)  # 


    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # 2x downsample for whisper


        encoder_outs = self.encoder(audio_mel.permute(0, 2, 1)).last_hidden_state # bs*seq*dim
        encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)

        input_ids = input_ids[:, 80:]

        inputs_embeds = self.llm.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat((encoder_outs, inputs_embeds), dim=1)

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask


        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,)
        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(input=model_outputs.logits, dim=-1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)

                
        return model_outputs, acc

    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            # max_length=kwargs.get("max_length", 200),
            max_new_tokens=kwargs.get("max_new_tokens", 150),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            no_repeat_ngram_size=3,
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return model_outputs
    
    


        






