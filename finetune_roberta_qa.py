import os
import logging
os.environ["WANDB_PROJECT"] = "RoBERTa_QA_Finetune"

import argparse
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
)

import torch

from utils import RobertaConfig, ExtractiveQAPreProcesing
from model import RobertaForQuestionAnswering

import warnings
warnings.filterwarnings("ignore")

def parse_arguments():

    parser = argparse.ArgumentParser(description="Wav2Vec2 Finetuning Arguments on Librispeech")

    ### Experiment Logging ###
    parser.add_argument(
        "--experiment_name",
        required=True,
        type=str
    )

    parser.add_argument(
        "--working_directory",
        required=True,
        type=str
    )

    parser.add_argument(
        "--path_to_cache_dir",
        help="Path to huggingface cache if different from default",
        default=None,
        type=str
    )

    parser.add_argument(
        "--num_train_epochs",
        help="Number of epochs you want to train for",
        default=3,
        type=int
    )

    parser.add_argument(
        "--save_steps",
        help="After how many steps do you want to log a checkpoint",
        default=500,
        type=int
    )

    parser.add_argument(
        "--eval_steps",
        help="After how many steps do you want to evaluate on eval data",
        default=500,
        type=int
    )

    parser.add_argument(
        "--logging_steps",
        help="After how many steps do you want to log to Weights and Biases (if installed)",
        default=500,
        type=int
    )

    parser.add_argument(
        "--warmup_steps",
        help="Number of learning rate warmup steps",
        default=100,
        type=int
    )

    ### Training Arguments ###

    parser.add_argument(
        "--per_device_batch_size",
        help="Batch size for every gradient accumulation steps",
        default=2,
        type=int
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="Number of gradient accumulation steps you want",
        default=2,
        type=int
    )

    parser.add_argument(
        "--learning_rate",
        help="Max learning rate that we warmup to",
        default=2e-5,
        type=float
    )

    parser.add_argument(
        "--weight_decay",
        help="Weight decay applied to model parameters during training",
        default=0.01,
        type=float
    )

    parser.add_argument(
        "--save_total_limit",
        help="Max number of checkpoints to save",
        default=4,
        type=int
    )

    ### Backbone Arguments ###
    parser.add_argument(
        "--huggingface_model_name",
        help="Name for pretrained RoBERTa backbone and Tokenizer",
        default="deepset/roberta-base-squad2",
        type=str
    )

    parser.add_argument(
        "--path_to_pretrained_backbone",
        help="Path to model weights stored from our pretraining to initialize the backbone",
        default=None,
        type=str
    )

    parser.add_argument(
        "--pretrained_backbone",
        help="Do you want want a `pretrained` backbone that we made (need to provide path_to_pretrained_backbone), \
            `pretrained_huggingface` backbone (then need huggingface_model_name), or `random` initialized backbone",
        choices=("pretrained", "pretrained_huggingface", "random"),
        type=str
    )
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default="roberta-base")

    args = parser.parse_args()

    return args


### Load Arguments ###
args = parse_arguments()

def load_tokenizer(model_name):
    try:
        return RobertaTokenizerFast.from_pretrained(model_name)
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise

def load_model(config):
    try:
        return RobertaForQuestionAnswering(config)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
logging.basicConfig(level=logging.INFO)
logging.info("----------Loading dataset and tokenizer----------")

### Load Tokenizer ###
tokenizer = RobertaTokenizerFast.from_pretrained(args.huggingface_model_name)

### Load Config ###



dataset = load_dataset("stanfordnlp/coqa")
processor = ExtractiveQAPreProcesing()
tokenized_squad = dataset.map(processor, batched=True, remove_columns=dataset["train"].column_names)

# print(tokenized_squad.column_names)
### Load Model ###
if args.resume_from_checkpoint is not None:
    config = RobertaConfig(pretrained_backbone=args.pretrained_backbone,
                           path_to_pretrained_weights=args.path_to_pretrained_backbone)
    model = RobertaForQuestionAnswering(config)
    model.load_state_dict(torch.load(f"{args.resume_from_checkpoint}/training_args.bin", map_location="cpu"))
else:
    config = RobertaConfig(pretrained_backbone=args.pretrained_backbone,
                           path_to_pretrained_weights=args.path_to_pretrained_backbone)
    model = RobertaForQuestionAnswering(config)



### Load Default Collator, We padded to longest length so no padding necessary ##

data_collator = DefaultDataCollator()

### Define Training Arguments ###
training_args = TrainingArguments(
    output_dir=os.path.join(args.working_directory, args.experiment_name),
    per_device_train_batch_size=args.per_device_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    # evaluation_strategy="steps",
    num_train_epochs=args.num_train_epochs,
    bf16=True,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    logging_steps=args.logging_steps,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    warmup_steps=args.warmup_steps,
    save_total_limit=args.save_total_limit,
    run_name=args.experiment_name,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

### TRAIN MODEL !!! ###
# trainer.train()
trainer.train(resume_from_checkpoint="model/RoBERTa/finetune_qa_hf_roberta_backbone/checkpoint-54324")

### Save Final Model ###
trainer.save_model("/home/tangsan/AllNlpProject/CoQAChat/model/RoBERTa/save_model")