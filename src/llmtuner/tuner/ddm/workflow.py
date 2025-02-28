# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py
import torch
from typing import TYPE_CHECKING, Optional, List, Union, Tuple
from transformers import DataCollatorForLanguageModeling, Seq2SeqTrainingArguments
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.core.metric import compute_acc
import torch.distributed as dist
from functools import partial

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DiffusionArguments, DataArguments, FinetuningArguments, GeneratingArguments

def run(
    trainer_cls,
    model_args: "ModelArguments",
    diffusion_args: "DiffusionArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, diffusion_args=diffusion_args)
    training_args_dict = training_args.to_dict()
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    
    dataset = get_dataset(model_args, data_args)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage=finetuning_args.stage)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    trainer = trainer_cls(
        diff_args=diffusion_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=partial(compute_acc, tokenizer=tokenizer, data_name=data_args.dataset),
        **split_dataset(dataset, data_args, training_args)
    )

    # Training
    if training_args.do_train:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        model.config.save_pretrained(training_args.output_dir)
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_acc"])
        
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
