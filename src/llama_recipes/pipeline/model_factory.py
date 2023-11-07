from llama_recipes.models.slam_model import setup_model, setup_tokenizer

def model_factory(train_config, model_config, **kwargs):

    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    model = setup_model(tokenizer, train_config, model_config, **kwargs)

    return model, tokenizer
