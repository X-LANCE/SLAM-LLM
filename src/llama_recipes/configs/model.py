from dataclasses import dataclass

    
@dataclass
class model_config:
    llm_name: str =  "llama-2-7b-hf"
    llm_path: str = "PATH/to/LLAMA/7B"
    encoder_name: str = None
    encoder_path: str = None
    encoder_projector: str = "linear"