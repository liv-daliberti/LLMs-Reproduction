#open_r1/utils/get_dataset.py
import logging
import datasets
from datasets import DatasetDict, concatenate_datasets

from transformers import PreTrainedTokenizer
from ..configs import ScriptArguments, SFTConfig
from trl import ModelConfig

logger = logging.getLogger(__name__)


def get_dataset(args: ScriptArguments, model_args: ModelConfig, training_args: SFTConfig) -> DatasetDict:
    """
    Load (or mix) the raw dataset(s), then tokenize both prompt + response columns
    so that each example has:
      {
        "input_ids": [...],
        "attention_mask": [...],
        "labels": [...],   # (with padding tokens set to -100)
      }

    Requires that:
      - args.dataset_prompt_column (e.g. "problem") is set
      - args.dataset_response_column (e.g. "solution") is set
      - args.max_seq_length is set
      - A tokenizer can be instantiated by get_tokenizer(args)
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}/{args.dataset_config or ''}")
        raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config)
    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        list_ds = []
        for ds_cfg in args.dataset_mixture.datasets:
            logger.info(
                f"  - Loading {ds_cfg.id}/{ds_cfg.config}, split={ds_cfg.split}"
                + (f", columns={ds_cfg.columns}" if ds_cfg.columns else "")
                + (f", weight={ds_cfg.weight}" if ds_cfg.weight else "")
            )
            ds = datasets.load_dataset(ds_cfg.id, ds_cfg.config, split=ds_cfg.split)
            if ds_cfg.columns is not None:
                ds = ds.select_columns(ds_cfg.columns)
            if ds_cfg.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * ds_cfg.weight)))
                logger.info(
                    f"    • Subsampled {ds_cfg.id}/{ds_cfg.config} → {len(ds)} examples"
                )
            list_ds.append(ds)

        if not list_ds:
            raise ValueError("No sub-datasets were loaded from the mixture configuration")

        combined = concatenate_datasets(list_ds).shuffle(seed=seed)
        logger.info(f"Combined mixture shape = {len(combined)} examples")

        if args.dataset_mixture.test_split_size is not None:
            split_dict = combined.train_test_split(
                test_size=args.dataset_mixture.test_split_size, seed=seed
            )
            logger.info(
                f"  • Split mixture into train/test with test_size={args.dataset_mixture.test_split_size}"
            )
            raw_datasets = DatasetDict({"train": split_dict["train"], "test": split_dict["test"]})
        else:
            raw_datasets = DatasetDict({"train": combined})
    else:
        raise ValueError("Either dataset_name or dataset_mixture must be provided")

    # At this point, raw_datasets is a DatasetDict with splits like "train", "test" (if present).
    # Next: we need to tokenize prompt+response.

    # --- Delayed import: only do this *inside* the function, after all other top-level imports ---
    from open_r1.utils.model_utils import get_tokenizer

    # 1) Instantiate the tokenizer (the same one sft.py uses)
    tokenizer: PreTrainedTokenizer = get_tokenizer(model_args, training_args)

    prompt_col = args.dataset_prompt_column
    resp_col   = args.dataset_response_column
    max_len    = training_args.max_seq_length

    if prompt_col is None or resp_col is None:
        raise ValueError(
            "dataset_prompt_column and dataset_response_column must both be set in ScriptArguments"
        )

    def preprocess(example):
        """
        For each raw example, build:
          inputs = tokenizer(prompt_text, truncation/padding)
          labels = tokenizer(response_text, truncation/padding → then mask pad→ -100)
          return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels_ids_with_-100_masking
          }
        """
        prompt_text = example[prompt_col]
        target_text = example[resp_col]

        # Tokenize the prompt
        tokenized_inputs = tokenizer(
            prompt_text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

        # Tokenize the target (for labels)
        tokenized_labels = tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

        label_ids = tokenized_labels["input_ids"]
        # Replace pad_token_id with -100 so CE loss ignores padding
        pad_id = tokenizer.pad_token_id
        label_ids = [(tid if tid != pad_id else -100) for tid in label_ids]

        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs

    # 2) Apply preprocess() to each split
    tokenized_splits = {}
    for split_name, split_ds in raw_datasets.items():
        logger.info(f"Tokenizing split '{split_name}' ({len(split_ds)} examples)…")
        tokenized = split_ds.map(
            preprocess,
            remove_columns=split_ds.column_names,  # drop all original columns
            batched=False,
        )
        tokenized_splits[split_name] = tokenized

    return DatasetDict(tokenized_splits)


