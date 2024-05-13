from slam_llm.utils.dataset_utils import load_module_from_py_file
from pathlib import Path

def get_custom_model_factory(model_config, logger):
    costom_model_path = model_config.get(
        "file", None
    )
    if costom_model_path is None:
        from slam_llm.models.slam_model import model_factory
        return model_factory

    if ":" in costom_model_path:
        module_path, func_name = costom_model_path.split(":")
    else:
        module_path, func_name = costom_model_path, "model_factory"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")
    
    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)
    except AttributeError as e:
        logger.info(f"It seems like the given method name ({func_name}) is not present in the model .py file ({module_path.as_posix()}).")
        raise e
    
