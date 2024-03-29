from slam_llm.pipeline.finetune_deepspeed import main as train

import hydra
import logging
import functools
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf
from asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig
from textwrap import dedent
from hydra import version
from hydra.main import _UNSPECIFIED_, _get_rerun_conf
from hydra._internal.deprecation_warning import deprecation_warning
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.types import TaskFunction
from hydra.core.utils import _flush_loggers, configure_log
from typing import Any, Callable, List, Optional


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    deepspeed_config: str = field(default="examples/asr/ds_config.json", metadata={"help": "The metric for evaluation"})

def main_wrapper(
    config_path: Optional[str] = _UNSPECIFIED_,
    config_name: Optional[str] = None,
    version_base: Optional[str] = _UNSPECIFIED_,
) -> Callable[[TaskFunction], Any]:
    """
    :param config_path: The config path, a directory where Hydra will search for
                        config files. This path is added to Hydra's searchpath.
                        Relative paths are interpreted relative to the declaring python
                        file. Alternatively, you can use the prefix `pkg://` to specify
                        a python package to add to the searchpath.
                        If config_path is None no directory is added to the Config search path.
    :param config_name: The name of the config (usually the file name without the .yaml extension)
    """

    version.setbase(version_base)

    if config_path is _UNSPECIFIED_:
        if version.base_at_least("1.2"):
            config_path = None
        elif version_base is _UNSPECIFIED_:
            url = "https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/changes_to_hydra_main_config_path"
            deprecation_warning(
                message=dedent(
                    f"""
                config_path is not specified in @hydra.main().
                See {url} for more information."""
                ),
                stacklevel=2,
            )
            config_path = "."
        else:
            config_path = "."

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args_parser = get_args_parser()
                args_parser.add_argument("--local_rank", type=int, default=-1)
                args = args_parser.parse_args()
                if args.experimental_rerun is not None:
                    cfg = _get_rerun_conf(args.experimental_rerun, args.overrides)
                    task_function(cfg)
                    _flush_loggers()
                else:
                    # no return value from run_hydra() as it may sometime actually run the task_function
                    # multiple times (--multirun)
                    _run_hydra(
                        args=args,
                        args_parser=args_parser,
                        task_function=task_function,
                        config_path=config_path,
                        config_name=config_name,
                    )

        return decorated_main

    return main_decorator


@main_wrapper(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    def to_plain_list(cfg_item):
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {k: to_plain_list(v) for k, v in cfg_item.items()}
        else:
            return cfg_item
    
    # kwargs = to_plain_list(cfg)
    kwargs = cfg
    log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
    
    logging.basicConfig(level=log_level)
    
    if kwargs.get("debug", False):
        import pdb;
        pdb.set_trace()
        
    train(kwargs)


if __name__ == "__main__":
    main_hydra()