import os
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union
from datasets import load_from_disk
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer
    from llmtuner.hparams import DataArguments


logger = get_logger(__name__)

def pad_sequence(lists, padding_value, cut_len):
    new_lists = []
    for l in lists:
        if len(l) >= cut_len:
            new_lists.append(l[:cut_len])
        else:
            new_lists.append(l+[padding_value]*(cut_len-len(l)))
    return new_lists
        
def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: str,
) -> Union["Dataset", "IterableDataset"]:

    def preprocess_s2s_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "src_mask": []}
    
        for src, tgt in zip(examples['prompt'], examples['response']):
            src_ids = tokenizer.encode(src)
            tgt_ids = tokenizer.encode(tgt)
            if data_args.cutoff_len is not None:
                tgt_ids = tgt_ids[:(data_args.cutoff_len-2)]
                src_ids = src_ids[-(data_args.cutoff_len-2-len(tgt_ids)):]
            
            input_ids = src_ids + [tokenizer.sep_token_id] + tgt_ids + [tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["src_mask"].append([1] * (len(src_ids) + 1))
            
        model_inputs["input_ids"] = pad_sequence(model_inputs["input_ids"], padding_value=tokenizer.pad_token_id, cut_len=data_args.cutoff_len)
        model_inputs["attention_mask"] = pad_sequence(model_inputs["attention_mask"], padding_value=1, cut_len=data_args.cutoff_len)
        model_inputs["src_mask"] = pad_sequence(model_inputs["src_mask"], padding_value=0, cut_len=data_args.cutoff_len)
        # print(model_inputs)
        return model_inputs

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for src, tgt in zip(examples['prompt'], examples['response']):
            src_ids = tokenizer.encode(src) + [tokenizer.sep_token_id]
            tgt_ids = tokenizer.encode(tgt) + [tokenizer.eos_token_id]
            if data_args.cutoff_len is not None:
                tgt_ids = tgt_ids[:(data_args.cutoff_len)]
                src_ids = src_ids[-(data_args.cutoff_len-len(tgt_ids)):]

            if data_args.train_on_prompt:
                source_mask = src_ids
            else:
                source_mask = [IGNORE_INDEX] * len(src_ids)
            
            labels = source_mask + tgt_ids
            input_ids = src_ids + tgt_ids 

            model_inputs["input_ids"].append(input_ids)
            # model_inputs["attention_mask"].append([1] * len(src_ids)+ [0]*(len(input_ids)-len(src_ids)))
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for src, tgt in zip(examples['prompt'], examples['response']):
            input_ids = tokenizer.encode(src) + [tokenizer.sep_token_id]
            # input_ids = tokenizer.encode(src)
            labels = tokenizer.encode(tgt)

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
            if len(labels) > data_args.cutoff_len:
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if stage != "sft":
        # diffusion training & inference
        preprocess_func = preprocess_s2s_dataset
        print_function = print_unsupervised_dataset_example
    elif stage == "sft" and not training_args.predict_with_generate:
        # AR training
        preprocess_func = preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    else:
        # AR inference
        preprocess_func = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    if data_args.cache_path is not None and os.path.exists(data_args.cache_path):
        logger.warning("Loading dataset from disk will ignore other data arguments.")
        return load_from_disk(data_args.cache_path)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        if data_args.cache_path is not None and not os.path.exists(data_args.cache_path):
            if training_args.should_save:
                dataset.save_to_disk(data_args.cache_path)
            raise SystemExit("Dataset saved, rerun this script with the same `--cache_file`.")

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Empty dataset!")

        return dataset
