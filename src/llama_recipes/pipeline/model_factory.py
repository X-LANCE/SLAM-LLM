from llama_recipes.models.slam_model import setup_model, setup_tokenizer
from llama_recipes.models.avsr_model import setupavsr_model

def model_factory(train_config, model_config, **kwargs):

    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    if model_config.name=="avsr":
        model = setupavsr_model(tokenizer, train_config, model_config, **kwargs).cuda()
    else:
        model = setup_model(tokenizer, train_config, model_config, **kwargs).cuda()

    return model, tokenizer
