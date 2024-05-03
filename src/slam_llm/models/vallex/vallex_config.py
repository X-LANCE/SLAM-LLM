from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from fairseq.data import Dictionary
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

logger = logging.get_logger(__name__)



class VallexConfig(PretrainedConfig):
    
    model_type = "vallex"
    
    def __init__(self,
            n_layer=24,
            n_head=16,
            n_dim=1024,
            prefix_mode=1,
            num_quantizers=8,
            sample_rate=24000,
            ar_at_dict="",
            ar_st_dict="",
            nar_at_dict="",
            nar_st_dict="",
            nar_scale_factor=1.0,
            prepend_bos=True,
            norm_first=True,
            eps=0.0,
            only_ar=False,
            only_nar=False,
            **kwargs
        ):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_dim = n_dim
        self.prefix_mode = prefix_mode
        self.num_quantizers = num_quantizers
        self.sample_rate = sample_rate
        self.nar_scale_factor = nar_scale_factor
        self.prepend_bos = prepend_bos
        self.norm_first = norm_first
        
        self.ar_at_dict = ar_at_dict
        self.ar_st_dict = ar_st_dict
        self.nar_at_dict = nar_at_dict
        self.nar_st_dict = nar_st_dict
        self.eps = eps
        self.only_ar = only_ar
        self.only_nar = only_nar
        
        super().__init__(
            **kwargs
        )
        

AutoConfig.register("vallex", VallexConfig)