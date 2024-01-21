from dataclasses import dataclass

    
@dataclass
class model_config:
    llm_name: str =  "llama-2-7b-hf"
    llm_path: str = "PATH/to/LLAMA/7B"
    llm_dim: int = 4096
    encoder_name: str = None
    encoder_ds_rate: int = 2
    encoder_path: str = None
    encoder_dim: int = 1280
    encoder_projector: str = "linear"
    encoder_projector_ds_rate: int = 5

    DMODEL: int = 512
    FRONTEND_DMODEL: int = 1024   #这个是专门指moco的
    TX_ATTENTION_HEADS: int = 8
    TX_NUM_LAYERS: int = 6
    PE_MAX_LENGTH: int = 500
    AUDIO_FEATURE_SIZE: int = 1024
    VIDEO_FEATURE_SIZE: int = 2048
    TX_FEEDFORWARD_DIM: int= 2048
    TX_DROPOUT: int = 0.1
    CHAR_NUM_CLASSES: int = 40

    WORD_NUM_CLASSES: int = 500
    FRAME_LENGTH: int = 29
    MOCO_FRONTEND_FILE: str = "/nfs/yangguanrou.ygr/AVSR/pretrain_model/moco_frontend.pt" #"/home/oss/yangguanrou.ygr/AVSR/pretrain_model/moco_frontend.pt"
    WAV2VEC_FILE: str = "/nfs/yangguanrou.ygr/AVSR/pretrain_model/wav2vec_vox_new.pt" #"/home/oss/yangguanrou.ygr/AVSR/pretrain_model/wav2vec_vox_new.pt"
    MAIN_REQ_INPUT_LENGTH: int = 80
    modal: str = "AV"
    TRAIN_LRS3_MODEL_FILE: str = "/nfs/yangguanrou.ygr/AVSR/train-step_0108-wer_0.058.ckpt"  # "/home/oss/yangguanrou.ygr/AVSR/train-step_0108-wer_0.058.ckpt"  #单一模态是这个
    TRAINED_AO_FILE : str = "/nfs/yangguanrou.ygr/AVSR/check/train-step_0604-wer_0.054.ckpt"  #"/home/oss/yangguanrou.ygr/AVSR/check/train-step_0604-wer_0.054.ckpt"
    TRAINED_VO_FILE: str = "/nfs/yangguanrou.ygr/AVSR/check/train-step_1191-wer_0.674.ckpt"  #"/home/oss/yangguanrou.ygr/AVSR/check/train-step_1191-wer_0.674.ckpt"


# class data:
#     def __init__(self):
#         self.modality="video"         
#         self.use_audio_normalise=False

class audio_backbone:
    def __init__(self):
        self.adim= 768
        self.aheads= 12
        self.eunits= 3072
        self.elayers= 12
        self.transformer_input_layer= "conv1d"
        self.dropout_rate= 0.1
        self.transformer_attn_dropout_rate= 0.1
        self.transformer_encoder_attn_layer_type= "rel_mha"
        self.macaron_style= True
        self.use_cnn_module= True
        self.cnn_module_kernel= 31
        self.zero_triu= False
        self.a_upsample_ratio= 1
        self.relu_type= "swish"
        self.ddim= 768
        self.dheads= 12
        self.dunits= 3072
        self.dlayers= 6
        self.lsm_weight= 0.1
        self.transformer_length_normalized_loss= False
        self.mtlalpha= 0.1
        self.ctc_type= "builtin"
        self.rel_pos_type= "latest"

class audiovisual_backbone:
    def __init__(self):                    
        self.adim= 768
        self.aheads= 12
        self.eunits= 3072
        self.elayers= 12
        self.transformer_input_layer= "conv3d"
        self.dropout_rate= 0.1
        self.transformer_attn_dropout_rate= 0.1
        self.transformer_encoder_attn_layer_type= "rel_mha"
        self.macaron_style= True
        self.use_cnn_module= True
        self.cnn_module_kernel= 31
        self.zero_triu= False
        self.a_upsample_ratio= 1
        self.relu_type= "swish"
        self.ddim= 768
        self.dheads= 12
        self.dunits= 3072
        self.dlayers= 6
        self.lsm_weight= 0.1
        self.transformer_length_normalized_loss= False
        self.mtlalpha= 0.1
        self.ctc_type= "builtin"
        self.rel_pos_type= "latest"

        self.aux_adim= 768
        self.aux_aheads= 12
        self.aux_eunits= 3072
        self.aux_elayers= 12
        self.aux_transformer_input_layer= "conv1d"
        self.aux_dropout_rate= 0.1
        self.aux_transformer_attn_dropout_rate= 0.1
        self.aux_transformer_encoder_attn_layer_type= "rel_mha"
        self.aux_macaron_style= True
        self.aux_use_cnn_module= True
        self.aux_cnn_module_kernel= 31
        self.aux_zero_triu= False
        self.aux_a_upsample_ratio= 1
        self.aux_relu_type= "swish"
        self.aux_dunits= 3072
        self.aux_dlayers= 6
        self.aux_lsm_weight= 0.1
        self.aux_transformer_length_normalized_loss= False
        self.aux_mtlalpha= 0.1
        self.ctc_type= "builtin"
        self.rel_pos_type= "latest"

        self.fusion_hdim= 8192
        self.fusion_norm= "batchnorm"


class visual_backbone:
    def __init__(self):
        self.adim= 768
        self.aheads= 12
        self.eunits= 3072
        self.elayers= 12
        self.transformer_input_layer= "conv3d"
        self.dropout_rate= 0.1
        self.transformer_attn_dropout_rate= 0.1
        self.transformer_encoder_attn_layer_type= "rel_mha"
        self.macaron_style= True
        self.use_cnn_module= True
        self.cnn_module_kernel= 31
        self.zero_triu= False
        self.a_upsample_ratio= 1
        self.relu_type= "swish"
        self.ddim= 768
        self.dheads= 12
        self.dunits= 3072
        self.dlayers= 6
        self.lsm_weight= 0.1
        self.transformer_length_normalized_loss= False
        self.mtlalpha= 0.1
        self.ctc_type= "builtin"
        self.rel_pos_type= "latest"


class model:
    def __init__(self):
        self.audio_backbone = audio_backbone()
        self.audiovisual_backbone = audiovisual_backbone()
        self.visual_backbone = visual_backbone()


@dataclass
class avmodel_config: 
    def __init__(self):
        self.pretrained_model_path="/nfs/yangguanrou.ygr/auto_avsr/avsr_trlrwlrs2lrs3vox2avsp_base.pth"
        self.modality="video"
        self.model = model()
        # self.data = data()


















