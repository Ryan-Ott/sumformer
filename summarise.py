"""
-Keep the emb_dim >= 512, higher = better
-Ways to reduce model: reduce depth (first), then emb, then heads, then context as last resort
-1080Ti cards don't have mixed precesion working well yet so don't bother
-Use weights and biases for logging: wandb
-Get a baseline eval of three methods (ROGUE, BERTScore, BLEU) + Bits/token (lower=better) for next meeting
-Possible way to reduce context: use sliding window in decoder for raw input (so not the latent space). Another would be to create a convolution in the encoder. (last resort)
-Batch size should not go lower than 16
-Important is throughput, so batch size should be as high as possible to maximise GPU utilisation (see on wandb) - this you can check in one epoch. LR can be checked in like half an hour (using -a on DAS5)
-Teacher forcing!!!
-Huggingface trainer class??

-Related work section: 1) summarisation, 2) transformers, 3) summarisation with transformers 4) cramming
-Paper instead of thesis: expert audience, no need to explain basics.
-The model that I use myself I need to explain in much detail to allow for reproducibility.
-Use conference paper style (NeurIPS perhaps?) - single column
"""
import math
import os
import torch
import multiprocessing
import fire
import wandb

from tokenizers import Tokenizer
from torch.optim import Adam, lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from utils import *
from modules import CollateFn
from transformer import Sumformer

# PARAMETERS
VOCAB_SIZE = 2**15
PATIENCE = 3
ACCUMULATION = 1
GRADIENT_CLIP = 1.0

def train(model, epochs, train_loader, val_loader, optimizer, scheduler, loss_fn, accumulation_steps, gradient_clip, log):
    best_val_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        print(f"Training epoch {epoch+1}")
        # ----- Training -----
        model.train()
        running_loss = 0

        for idx, batch in enumerate(train_loader):
            if idx % 1000 == 0: print(f"Batch {idx}")
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            targets = batch["targets"].to(model.device)

            optimizer.zero_grad()
            
            output = model(input_ids, targets, doc_mask=attention_mask)
            outputs = output.view(-1, output.size(-1))  # flatten the output to be (batch_size * max_len, vocab_size) from (batch_size, max_len, vocab_size)
            targets = targets.view(-1)  # flatten the targets to be (batch_size * max_len) from (batch_size, max_len)
           
            loss = loss_fn(outputs, targets)
            loss.backward()

            # Gradient accumulation
            if accumulation_steps > 1:
                if (idx + 1) % accumulation_steps == 0:
                    clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                    scheduler.step()
            else:
                clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()
            
            running_loss += loss.item()

        epoch_train_loss = running_loss + len(train_loader)
        if log: wandb.log({"train_loss": epoch_train_loss})
        # ----- Validation -----
        model.eval()
        running_loss = 0

        for idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            targets = batch["targets"].to(model.device)

            with torch.no_grad():
                output = model(input_ids, targets, doc_mask=attention_mask)
                outputs = output.view(-1, output.size(-1))
                targets = targets.view(-1)

            loss = loss_fn(outputs, targets)
            running_loss += loss.item()
        
        epoch_val_loss = running_loss + len(val_loader)
        if log: wandb.log({"val_loss": epoch_val_loss})

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1} | Training loss: {epoch_train_loss} | Validation loss: {epoch_val_loss}")
        # ----- Early Stopping -----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience = 0
            torch.save(model.state_dict(), "model.pt")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping - validation loss has not improved in {PATIENCE} epochs.")
                break



def main(epochs=1, alpha=3e-4, batch_size=16, emb_dim=512, sched="onecycle", e_heads=4, e_hidden=4, e_depth=3, d_heads=4, d_hidden=2, d_depth=8, debug=False, log=False):
    num_proc = max(multiprocessing.cpu_count()-2, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log:
        wandb.init(project="Sumformer", entity="ryanott")
        wandb.config.update({"epochs": epochs, "alpha": alpha, "batch_size": batch_size, "emb_dim": emb_dim, "sched": sched, "e_heads": e_heads, "e_hidden": e_hidden, "e_depth": e_depth, "d_heads": d_heads, "d_hidden": d_hidden, "d_depth": d_depth})

    print("\n>>Loading data...")
    train_data, val_data, test_data = load_reddit(train_split=0.8, val_split=0.1)

    if not os.path.exists("data/tokenizer.json"):
        print("\n>>Training new tokenizer...")
        train_tokenizer(VOCAB_SIZE, train_data['document'])
    print("\n>>Loading tokenizer...")
    tokenizer = Tokenizer.from_file("data/tokenizer.json")

    if debug:
        print("CHEKING TOKENIZER TRAINING")
        test_sentence_1 = "The quick brown fox jumps over the lazy dog."
        test_sentence_2 = "A fat cat sat on the mat."
        encoded_test = tokenizer.encode(test_sentence_1, test_sentence_2)
        print(f"Original sentences: {test_sentence_1}, {test_sentence_2}")
        print(f"Encoded sentences: {encoded_test.tokens}")
    
    print("\n>>Creating data loaders...")
    train_data = sort_by_length(train_data)
    val_data = sort_by_length(val_data)
    test_data = sort_by_length(test_data)

    collate_fn = CollateFn(tokenizer)

    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn.collate)#, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn.collate)#, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn.collate)#, num_workers=1)

    if debug:
        print("CHECKING ENCODING")
        first_batch = next(iter(train_loader))
        docs = first_batch["input_ids"]
        sums = first_batch["targets"]
        print("Documents Shape:", docs.shape)
        print("Summaries Shape:", sums.shape)
        print("Explanation:")
        print(" - Batch size:", docs.shape[0])  # Number of instances in the batch
        print(" - Maximum sequence length of documents:", docs.shape[1])  # Maximum length of documents in the batch
        print(" - Maximum sequence length of summaries:", sums.shape[1])  # Maximum length of summaries in the batch
        # printing the last instance from first_batch
        last_doc = first_batch["input_ids"][-1].tolist()
        last_mask = first_batch["attention_mask"][-1].tolist()
        last_sum = first_batch["targets"][-1].tolist()
        decoded_doc = tokenizer.decode(last_doc, skip_special_tokens=False)
        decoded_sum = tokenizer.decode(last_sum, skip_special_tokens=False)
        print(" - Last Document in First Batch:", decoded_doc)
        print(" - Last Attention Mask in First Batch:", last_mask)
        print(" - Last Summary in First Batch:", decoded_sum)

    print("\n>>Creating model...")
    model = Sumformer(
        vocab_size=tokenizer.get_vocab_size(),
        max_input_len=find_longest_instance([train_loader, val_loader, test_loader]),
        emb_dim=emb_dim,
        e_heads=e_heads,
        e_hidden=e_hidden*emb_dim,
        e_depth=e_depth,
        d_heads=d_heads,
        d_hidden=d_hidden*emb_dim,
        d_depth=d_depth).to(device)
    
    print("\n>>Creating optimizer, scheduler and loss function...")
    optimizer = Adam(model.parameters(), lr=alpha)

    lr_schedules = {
        "linear": lr_scheduler.LinearLR(optimizer, start_factor=alpha/10, end_factor=alpha, total_iters=len(train_loader)*epochs),
        "constant": lr_scheduler.ConstantLR(optimizer),
        "onecycle": lr_scheduler.OneCycleLR(optimizer, max_lr=alpha, total_steps=len(train_loader)*epochs, pct_start=0.3, anneal_strategy="linear"),
        "invsqrt": lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/math.sqrt(epoch) if epoch > 0 else 1),
        "cosinedecay": lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)
    }

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    print("\n>>Training...")
    train(model, epochs, train_loader, val_loader, optimizer, lr_schedules[sched], loss_fn, ACCUMULATION, GRADIENT_CLIP, log)

    print("\n>>Training complete...")


if __name__ == "__main__":
    fire.Fire(main)
