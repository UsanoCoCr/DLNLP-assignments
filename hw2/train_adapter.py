import logging
from datasets import load_dataset
from dataclasses import dataclass, field, asdict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorWithPadding,
    AdapterConfig
)
from transformers import set_seed
import numpy as np
import torch
import wandb
import os
from typing import Optional
from sklearn.metrics import f1_score, accuracy_score
from adapters import AutoAdapterModel

from dataHelper import get_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
os.environ["WANDB_PROJECT"]="NLP_hw2_task2"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define data classes for arguments
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    num_labels: int = field(default=13, metadata={"help": "Number of labels for classification"})

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(metadata={"help": "Name of the dataset"})
    max_seq_length: int = field(default=128, metadata={"help": "The maximum total input sequence length after tokenization."})

@dataclass
class TrainingConfig:
    output_dir: str = field(default="./results", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    overwrite_output_dir: bool = field(default=True, metadata={"help": "Overwrite the content of the output directory"})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per device during training"})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size for evaluation"})
    num_train_epochs: float = field(default=5.0, metadata={"help": "Total number of training epochs to perform."})
    # seed: 42, 20, 11, 5, 2
    seed: int = field(default=42, metadata={"help": "Random seed for initialization"})
    report_to: str = field(default="wandb", metadata={"help": "Use Weights & Biases for tracking"})
    run_name: str = field(default="bert_res_run1", metadata={"help": "Name of this WandB run"})
    # run_name: str = field(default="roberta_base_run1", metadata={"help": "Name of this WandB run"})
    # run_name: str = field(default="scibert_base_run1", metadata={"help": "Name of this WandB run"})
    save_total_limit: int = field(default=1, metadata={"help": "Limit the total amount of checkpoints."})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Whether or not to load the best model found during training at the end."})
    metric_for_best_model: str = field(default="eval_loss", metadata={"help": "The metric to use to compare two different models."})
    evaluation_strategy: str = field(default="steps", metadata={"help": "The evaluation strategy to adopt during training."})
    save_strategy: str = field(default="steps", metadata={"help": "The checkpoint save strategy to adopt during training."})

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)

    metrics = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy
    }
    return metrics

# initialize logging, seed, argparse...
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingConfig))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
set_seed(training_args.seed)
wandb.init(project="NLP_hw2_task2", name="Bert_res_run1")
# wandb.init(project="NLP_hw2_task2", name="RoBerta_base_run1")
# wandb.init(project="NLP_hw2_task2", name="SciBert_base_run1")

# load datasets
print(f"Loading dataset {data_args.dataset_name}")
dataset = get_dataset(data_args.dataset_name, '<sep>')

# load models
print(f"Loading model {model_args.model_name_or_path}")
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, local_files_only=True)
config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=model_args.num_labels, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config, local_files_only=True)
adapter_config = AdapterConfig.load("pfeiffer", non_linearity="gelu", reduction_factor=16)
model.load_adapter("AdapterHub/bert-base-uncased-pf-imdb", config=adapter_config)
model.active_adapters = "bert-base-uncased-pf-imdb"
for name, param in model.named_parameters():
    if not param.data.is_contiguous():
        param.data = param.data.contiguous()
model = model.to(device)

# process datasets and build up datacollator
print("Preprocessing datasets")
def preprocess_function(examples):
    return tokenizer(examples['text'], max_length=data_args.max_seq_length, truncation=True, padding="max_length")
dataset["train"] = dataset["train"].map(preprocess_function, batched=True)
dataset["test"] = dataset["test"].map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# train the model
trainer = Trainer(
    model=model,
    args=TrainingArguments(**asdict(training_args)),
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()
wandb.finish()