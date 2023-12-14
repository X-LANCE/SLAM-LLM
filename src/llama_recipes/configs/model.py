from dataclasses import dataclass

    
@dataclass
class model_config:
    llm_name: str =  "llama-2-7b-hf"
    llm_path: str = "PATH/to/LLAMA/7B"
    encoder_name: str = None
    encoder_ds_rate: int = 2
    encoder_path: str = None
    encoder_projector: str = "linear"
    encoder_projector_ds_rate: int = 5

    # name: str =  "avsr"
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








