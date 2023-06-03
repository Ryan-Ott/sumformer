import matplotlib.pyplot as plt
import os
import torch

from datasets import Dataset, concatenate_datasets, load_dataset

from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def test(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for docs, sums in test_loader:
            docs = docs.to(device)
            sums = sums.to(device)

            output = model(docs)  # !! Change this

            loss = loss_fn(output.view(-1, output.size(-1)), sums.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)

    print(f"\nTest Loss: {avg_loss}")


def split_data(dataset, train_split, val_split):
    # Shuffle the dataset
    dataset = dataset.shuffle()
    dataset = dataset.flatten_indices()  # rewrite the shuffled dataset on disk again as contiguous chunks for speed

    # Split the dataset
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * val_split)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]

    return train_dataset, val_dataset, test_dataset


def load_reddit(train_split, val_split, min_len=30):
    """Concatenate the short and long reddit TIFU datasets and split into train, validation, and test sets. Keep only the docs and their summary."""
    dataset_short = load_dataset("reddit_tifu", "short")
    dataset_short = dataset_short.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'tldr'])
    dataset_short = dataset_short.rename_columns({'documents': 'document', 'title': 'summary'})

    dataset_long = load_dataset("reddit_tifu", "long")
    dataset_long = dataset_long.remove_columns(['ups', 'num_comments', 'upvote_ratio', 'score', 'title'])
    dataset_long = dataset_long.rename_columns({'documents': 'document', 'tldr': 'summary'})

    dataset = concatenate_datasets([dataset_short["train"], dataset_long["train"]])

    # Filtering out too short documents and summaries
    dataset = dataset.filter(lambda x:
                             len(x["document"]) > min_len
                             and len(x["summary"]) > min_len
                             and len(x["document"]) > len(x["summary"]))

    # Split the dataset into train, validation, and test sets after shuffling
    train_dataset, val_dataset, test_dataset = split_data(dataset, train_split, val_split)

    # Turn the dictionaries into huggingface datasets
    train_dataset = Dataset.from_dict(train_dataset)
    val_dataset = Dataset.from_dict(val_dataset)
    test_dataset = Dataset.from_dict(test_dataset)

    return train_dataset, val_dataset, test_dataset


def train_tokenizer(vocab_size, training_texts):
    """Train a new WordPiece tokenizer and save it to disk."""
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[SOS]", "[EOS]"]
    tokenizer.add_tokens(special_tokens)
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    tokenizer.train_from_iterator(training_texts, trainer)
    
    tokenizer.post_processor = TemplateProcessing(
    single = "[SOS]:1 $A:1 [EOS]:1",
    pair = "[SOS]:1 $A:1 [SEP]:1 $B:1 [EOS]:1",
    special_tokens = [
        ("[SOS]", tokenizer.get_vocab()["[SOS]"]),
        ("[EOS]", tokenizer.get_vocab()["[EOS]"]),
        ("[SEP]", tokenizer.get_vocab()["[SEP]"])]
    )
    if not os.path.exists("data"):
        os.mkdir("data")
    tokenizer.save("data/tokenizer.json")


def sort_by_length(dataset: Dataset):
    """Add a field length for the len of the document and sort the dataset by length."""
    dataset = dataset.map(lambda x: {'length': len(x['document'])})
    return dataset.sort('length')


def find_longest_instance(data_loaders):
    max_length = 0

    for data_loader in data_loaders:
        for batch in data_loader:
            # Assuming the input and target sequences are named "input_ids" and "targets"
            input_lengths = [len(seq) for seq in batch["input_ids"]]
            target_lengths = [len(seq) for seq in batch["targets"]]
            
            max_length = max(max_length, max(input_lengths), max(target_lengths))

    return max_length


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)