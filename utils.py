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



def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, clip, early_stopping, accumulation_steps=8):
    # Ensure 'graphs' directory exists
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    train_losses = []
    val_losses = []

    scaler = GradScaler()  # for mixed precision training

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0

        # For storing norms
        l2_norms = []

        optimizer.zero_grad()  # reset gradients at the beginning of each epoch

        for batch_idx, (docs, sums) in enumerate(train_loader):
            docs = docs.to(device)
            sums = sums.to(device)
            
            # Forward pass
            with autocast():  # enable mixed precision training
                output = model(docs, sums)
                loss = loss_fn(output.view(-1, output.size(-1)), sums.view(-1))

            loss = loss / accumulation_steps  # normalize the loss

            # Backward pass and optimization
            scaler.scale(loss).backward()  # scale the loss to prevent underflow/overflow

            # Compute L2 norm of gradients and store
            l2_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    l2_norm += param_norm.item() ** 2
            l2_norms.append(l2_norm ** 0.5)

            # Gradient clipping
            # Note: we don't need to scale the gradients here because scaler.step() will do it for us
            clip_grad_norm_(model.parameters(), clip)

            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:  # update weights every accumulation_steps
                scaler.step(optimizer)  # unscale the gradients before performing optimizer.step()
                scaler.update()
                optimizer.zero_grad()  # reset gradients

            # Update learning rate
            scheduler.step()

            running_loss += loss.item() * accumulation_steps  # scale the loss back up for reporting

            if batch_idx % 1000 == 0:
                print(f"Epoch {epoch} batch {batch_idx} loss {loss.item()}")

        running_loss /= len(train_loader)
        train_losses.append(running_loss)
        print(f"Epoch {epoch} Training Loss: {running_loss}")

        # After each epoch, validate the model
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for docs, sums in val_loader:
                docs = docs.to(device)
                sums = sums.to(device)
                
                with autocast():  # enable mixed precision training
                    output = model(docs, sums)
                    loss = loss_fn(output.view(-1, output.size(-1)), sums.view(-1))

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Validation loss after epoch {epoch}: {val_loss}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Plot L2 Norm
        plt.figure()
        plt.plot(l2_norms)
        plt.title('Gradient L2 Norm')
        plt.savefig(f'graphs/l2_norm_epoch_{epoch}.png')
        plt.close()

    print("Training completed.")

    # Plot Losses
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss curves')
    plt.legend()
    plt.savefig('graphs/loss_curves.png')
    plt.close()

    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt', map_location=device))

    # Plot weights and biases
    for name, param in model.named_parameters():
        if param.requires_grad:
            plt.figure()
            plt.hist(param.data.cpu().numpy().flatten(), bins=100)
            plt.title(name)
            plt.savefig(f'graphs/{name.replace(".", "_")}.png')
            plt.close()


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