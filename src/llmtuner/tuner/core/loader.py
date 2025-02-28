import os
import torch
from typing import TYPE_CHECKING, Optional, Tuple

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import importlib
try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError: # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import count_parameters, infer_optim_dtype
from llmtuner.hparams import FinetuningArguments
from llmtuner.tuner.core.adapter import init_adapter
from llmtuner.tuner.core.utils import prepare_model_for_training
from llmtuner.tuner.core.custom_tokenizer import CustomTokenizer
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from llmtuner.hparams import ModelArguments, DiffusionArguments


logger = get_logger(__name__)


check_min_version("4.31.0")
require_version("datasets>=2.12.0", "To fix: pip install datasets>=2.12.0")
require_version("accelerate>=0.21.0", "To fix: pip install accelerate>=0.21.0")
require_version("peft>=0.4.0", "To fix: pip install peft>=0.4.0")


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    diffusion_args: Optional["DiffusionArguments"] = None,
    stage = None
) -> Tuple[PreTrainedModel, "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None
    }
    model_to_load = model_args.model_name_or_path if model_args.checkpoint_dir is None else model_args.checkpoint_dir[0]

    tokenizer = CustomTokenizer.from_pretrained(model_to_load)
    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)
 
    # Set model dtype
    if model_args.compute_dtype is not None: # for training
        setattr(config, "torch_dtype", model_args.compute_dtype)
    else: # for evaluation, priority: bf16 > fp16 > fp32
        model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    # Quantization configurations (using bitsandbytes library).
    is_mergeable = True
    if model_args.quantization_bit is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))} if is_trainable else "auto"
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    stage = finetuning_args.stage if stage is None else stage
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            config=config,
            torch_dtype=model_args.compute_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs
        )
        logger.info(f"Loading pretrained model from {model_to_load}")
    except:
        model = AutoModelForCausalLM.from_config(
            config,
        )
        logger.info("Training from scratch...")
    # Load and prepare pre-trained models (without valuehead).
    if stage != 'sft':
        model_lib = importlib.import_module(f"llmtuner.tuner.{stage}.model")
        model = model_lib.DiffusionModel(model, config, diffusion_args)

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # Initialize adapters
    if stage == 'sft':
        model = prepare_model_for_training(model=model, finetuning_args=finetuning_args) if is_trainable else model
        model = init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)
        model = model.train() if is_trainable else model.eval()
    else:
        ## TODO: need to use model.model due to the wrap up
        model.denoise_model = prepare_model_for_training(model=model.denoise_model, finetuning_args=finetuning_args) if is_trainable else model.denoise_model
        checkpoint_dir = model_args.checkpoint_dir
        model_args.checkpoint_dir = None
        model.denoise_model = init_adapter(model.denoise_model, model_args, finetuning_args, is_trainable, is_mergeable)
        if checkpoint_dir is not None: # for sampling
            load_path = os.path.join(checkpoint_dir[0], 'pytorch_model.bin')
            loaded = torch.load(
                load_path,
                map_location=torch.device('cuda')
            )
            # print(loaded.keys())
            # try:
            model.load_state_dict(loaded, strict=False)
            # except:
                # model.model.load_state_dict(loaded)
            logger.info(f"Loading pretrained model from {load_path}")
        model = model.train() if is_trainable else model.eval()
        
    # Prepare model for inference
    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.to(model_args.compute_dtype) if model_args.quantization_bit is None else model

    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    return model, tokenizer