import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import math
import ast
import time
import random
import json

# Hyperparameters
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 20
MAX_LR = 1e-3
WEIGHT_DECAY = 0.01
CLIP = 1.0
LABEL_SMOOTHING = 0.1
BEAM_WIDTH = 5
MAX_PRED_LEN = 100

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


class Vocabulary:
    def __init__(self, special_tokens):
        self.token_to_idx = {tok: i for i, tok in enumerate(special_tokens)}
        self.idx_to_token = {i: tok for i, tok in enumerate(special_tokens)}
        self.n_tokens = len(special_tokens)
        self.PAD_IDX = self.token_to_idx["<PAD>"]
        self.SOS_IDX = self.token_to_idx["<SOS>"]
        self.EOS_IDX = self.token_to_idx["<EOS>"]
        self.UNK_IDX = self.token_to_idx["<UNK>"]

    def build_vocab(self, dataframe):
        all_tokens = set()
        for inp_str in dataframe["input_sequence"]:
            all_tokens.update(ast.literal_eval(inp_str))
        for out_str in dataframe["output_path"]:
            all_tokens.update(ast.literal_eval(out_str))

        for token in sorted(list(all_tokens)):
            if token not in self.token_to_idx:
                self.token_to_idx[token] = self.n_tokens
                self.idx_to_token[self.n_tokens] = token
                self.n_tokens += 1

    def numericalize(self, token_list, is_target=False):
        indices = [self.token_to_idx.get(token, self.UNK_IDX) for token in token_list]
        return [self.SOS_IDX] + indices + [self.EOS_IDX] if is_target else indices

    def denumericalize(self, index_list):
        return [self.idx_to_token.get(idx, "<UNK>") for idx in index_list]


class MazeDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.data = dataframe.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_tokens = ast.literal_eval(row["input_sequence"])
        path_tokens = ast.literal_eval(row["output_path"])
        src = self.vocab.numericalize(input_tokens, is_target=False)
        trg = self.vocab.numericalize(path_tokens, is_target=True)
        return torch.LongTensor(src), torch.LongTensor(trg)


def collate_fn(batch, pad_idx):
    srcs, trgs = zip(*batch)

    src_lens = [len(s) for s in srcs]
    src_max_len = max(src_lens)
    padded_srcs = torch.full((src_max_len, len(srcs)), pad_idx, dtype=torch.long)
    for i, src in enumerate(srcs):
        padded_srcs[: len(src), i] = src

    trg_lens = [len(t) for t in trgs]
    trg_max_len = max(trg_lens)
    padded_trgs = torch.full((trg_max_len, len(trgs)), pad_idx, dtype=torch.long)
    for i, trg in enumerate(trgs):
        padded_trgs[: len(trg), i] = trg

    return padded_srcs, padded_trgs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        divterm = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000.0) / d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(pos * divterm)
        pos_encoding[:, 1::2] = torch.cos(pos * divterm)
        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(0)
        pe = self.pos_encoding[:, :seq_len, :].permute(1, 0, 2)
        return x + pe


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, pad_idx
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(input_dim, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, input_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz), 1).bool().to(device)

    def _create_padding_mask(self, src):
        return (src == self.pad_idx).transpose(0, 1)

    def forward(self, src, trg):
        device = src.device
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_encoder(self.embedding(trg) * math.sqrt(self.d_model))

        trg_mask = self._generate_square_subsequent_mask(trg.shape[0], device)
        src_padding_mask = self._create_padding_mask(src)

        memory = self.transformer_encoder(
            src_emb, mask=None, src_key_padding_mask=src_padding_mask
        )
        output = self.transformer_decoder(
            trg_emb, memory, tgt_mask=trg_mask, memory_key_padding_mask=src_padding_mask
        )
        return self.fc_out(output)


def beam_search_decode(model, src, vocab, beam_width, max_len):
    model.eval()
    device = src.device

    if src.ndim == 1:
        src = src.unsqueeze(-1)

    with torch.no_grad():
        src_emb = model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model))
        src_padding_mask = model._create_padding_mask(src)
        memory = model.transformer_encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )

        # Initialize beam: (sequence, score)
        beams = [([vocab.SOS_IDX], 0.0)]

        for _ in range(max_len):
            candidates = []

            for seq, score in beams:
                if seq[-1] == vocab.EOS_IDX:
                    candidates.append((seq, score))
                    continue

                trg_tensor = torch.LongTensor(seq).unsqueeze(-1).to(device)
                trg_emb = model.pos_encoder(
                    model.embedding(trg_tensor) * math.sqrt(model.d_model)
                )
                trg_mask = model._generate_square_subsequent_mask(
                    trg_tensor.shape[0], device
                )

                output = model.transformer_decoder(
                    trg_emb,
                    memory,
                    tgt_mask=trg_mask,
                    memory_key_padding_mask=src_padding_mask,
                )
                logits = model.fc_out(output[-1, :, :].squeeze(0))
                log_probs = torch.log_softmax(logits, dim=-1)

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for log_prob, idx in zip(topk_log_probs, topk_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score))

            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            if all(seq[-1] == vocab.EOS_IDX for seq, _ in beams):
                break

        best_seq = beams[0][0]
        predicted_tokens = vocab.denumericalize(best_seq)

        if predicted_tokens[0] == "<SOS>":
            predicted_tokens.pop(0)
        if predicted_tokens and predicted_tokens[-1] == "<EOS>":
            predicted_tokens.pop(-1)

        return predicted_tokens


def train(model, iterator, optimizer, scheduler, criterion, clip):
    model.train()
    epoch_loss = 0
    device = next(model.parameters()).device

    for src, trg in iterator:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        trg_input = trg[:-1, :]
        trg_target = trg[1:, :]
        output = model(src, trg_input)

        output_dim = output.shape[-1]
        loss = criterion(output.reshape(-1, output_dim), trg_target.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, vocab, use_beam_search=False):
    model.eval()
    epoch_loss = 0
    sequence_correct = 0
    total_sequences = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            trg_input = trg[:-1, :]
            trg_target = trg[1:, :]

            output = model(src, trg_input)
            output_dim = output.shape[-1]
            loss = criterion(output.reshape(-1, output_dim), trg_target.reshape(-1))
            epoch_loss += loss.item()

            if use_beam_search:
                for i in range(src.shape[1]):
                    predicted = beam_search_decode(
                        model, src[:, i], vocab, BEAM_WIDTH, MAX_PRED_LEN
                    )
                    true_tokens = vocab.denumericalize(trg[:, i].tolist())
                    true_tokens = [
                        t for t in true_tokens if t not in ["<SOS>", "<EOS>", "<PAD>"]
                    ]

                    if predicted == true_tokens:
                        sequence_correct += 1
                    total_sequences += 1
            else:
                predicted_tokens = output.argmax(2)
                mask = trg_target != vocab.PAD_IDX
                correct_tokens = (predicted_tokens == trg_target) & mask
                is_correct_seq = correct_tokens.sum(0) == mask.sum(0)
                sequence_correct += is_correct_seq.sum().item()
                total_sequences += trg.shape[1]

    avg_loss = epoch_loss / len(iterator)
    seq_acc = sequence_correct / total_sequences if total_sequences > 0 else 0
    return avg_loss, seq_acc


def plot_metrics(train_losses, val_losses, val_accs, save_prefix):
    """Plot training and validation metrics"""
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(
        epochs_range,
        train_losses,
        "b-",
        marker="o",
        label="Train Loss",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        epochs_range,
        val_losses,
        "r-",
        marker="s",
        label="Val Loss",
        linewidth=2,
        markersize=6,
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs_range,
        val_accs,
        "g-",
        marker="D",
        label="Val Accuracy",
        linewidth=2,
        markersize=6,
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Validation Accuracy", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_training_metrics.png", dpi=300, bbox_inches="tight")
    print(f"Training metrics plot saved to: {save_prefix}_training_metrics.png")
    plt.close()


def plot_test_accuracy(test_acc, val_acc, save_prefix):
    """Plot comparison of validation and test accuracy"""
    plt.figure(figsize=(8, 6))

    accuracies = [val_acc, test_acc]
    labels = ["Validation", "Test"]
    colors = ["#3498db", "#2ecc71"]

    bars = plt.bar(
        labels, accuracies, color=colors, alpha=0.7, edgecolor="black", linewidth=2
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}\n({height*100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Model Performance: Validation vs Test", fontsize=14, fontweight="bold")
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_test_accuracy.png", dpi=300, bbox_inches="tight")
    print(f"Test accuracy plot saved to: {save_prefix}_test_accuracy.png")
    plt.close()


def evaluate_and_sample_test(
    model, test_data, vocab, device, num_samples=5, save_prefix="transformer"
):
    """
    Evaluate model on test set and return random samples for visualization
    Returns: test_accuracy, list of sample predictions
    """
    model.eval()
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)

    all_predictions = []
    correct = 0
    total = len(test_data)

    # Get predictions for all test samples
    for idx in range(total):
        src, trg = test_data[idx]
        src = src.to(device)

        # Get prediction
        predicted_path = beam_search_decode(model, src, vocab, BEAM_WIDTH, MAX_PRED_LEN)

        # Get true path
        true_tokens = vocab.denumericalize(trg.tolist())
        true_tokens = [t for t in true_tokens if t not in ["<SOS>", "<EOS>", "<PAD>"]]

        # Get input sequence
        input_tokens = vocab.denumericalize(src.tolist())

        # Check if correct
        is_correct = predicted_path == true_tokens
        if is_correct:
            correct += 1

        all_predictions.append(
            {
                "index": idx,
                "input_sequence": input_tokens,
                "true_path": true_tokens,
                "predicted_path": predicted_path,
                "correct": is_correct,
            }
        )

        if (idx + 1) % 100 == 0:
            print(
                f"Processed {idx + 1}/{total} samples... Current Acc: {correct/(idx+1):.4f}"
            )

    test_accuracy = correct / total
    print(f"\n{'='*80}")
    print(
        f"FINAL TEST ACCURACY: {test_accuracy:.4f} ({correct}/{total}) = {test_accuracy*100:.2f}%"
    )
    print(f"{'='*80}\n")

    # Select random samples
    random_samples = random.sample(
        all_predictions, min(num_samples, len(all_predictions))
    )

    # Save samples to file
    samples_file = f"{save_prefix}_test_samples.json"
    with open(samples_file, "w") as f:
        json.dump(random_samples, f, indent=2)
    print(f"Sample predictions saved to: {samples_file}")

    return test_accuracy, random_samples


def print_sample_predictions(samples):
    """Print sample predictions in a formatted way"""
    print("\n" + "=" * 80)
    print("RANDOM TEST SAMPLE PREDICTIONS")
    print("=" * 80)

    for i, sample in enumerate(samples, 1):
        print(f"\n{'─'*80}")
        print(f"Sample {i} (Test Index: {sample['index']})")
        print(f"{'─'*80}")
        print(f"Input Sequence (first 10 tokens): {sample['input_sequence'][:10]}...")
        print(f"\nTrue Path:      {sample['true_path']}")
        print(f"Predicted Path: {sample['predicted_path']}")
        print(f"\nMatch: {'✓ CORRECT' if sample['correct'] else '✗ INCORRECT'}")
        print(f"{'─'*80}")

    print("\n" + "=" * 80)


def save_samples_for_visualization(samples, save_prefix="transformer"):
    """
    Save samples in a format that can be easily used with your visualization utility
    Each sample saved as a separate txt file with input and output
    """
    import os

    viz_dir = f"{save_prefix}_visualization_samples"
    os.makedirs(viz_dir, exist_ok=True)

    for i, sample in enumerate(samples, 1):
        # Save input
        input_file = os.path.join(viz_dir, f"sample_{i}_input.txt")
        with open(input_file, "w") as f:
            f.write(str(sample["input_sequence"]))

        # Save predicted output
        pred_file = os.path.join(viz_dir, f"sample_{i}_predicted_output.txt")
        with open(pred_file, "w") as f:
            f.write(str(sample["predicted_path"]))

        # Save true output
        true_file = os.path.join(viz_dir, f"sample_{i}_true_output.txt")
        with open(true_file, "w") as f:
            f.write(str(sample["true_path"]))

        # Save summary
        summary_file = os.path.join(viz_dir, f"sample_{i}_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"Sample {i} - Test Index: {sample['index']}\n")
            f.write(f"Correct: {sample['correct']}\n\n")
            f.write(f"Input: {sample['input_sequence']}\n\n")
            f.write(f"True Path: {sample['true_path']}\n\n")
            f.write(f"Predicted Path: {sample['predicted_path']}\n")

    print(f"\nVisualization files saved to: {viz_dir}/")
    print(f"You can now use your visualization utility on these files!")


def save_training_history(train_losses, val_losses, val_accs, save_prefix):
    """Save training history to JSON for later plotting"""
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accs,
        "epochs": len(train_losses),
    }

    history_file = f"{save_prefix}_training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_file}")


def main(train_data_path, test_data_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split data
    df = pd.read_csv(train_data_path)
    # Curriculum learning: sort by path length
    df["path_length"] = df["output_path"].apply(lambda x: len(ast.literal_eval(x)))
    df = df.sort_values("path_length").reset_index(drop=True)

    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)
    test_df = pd.read_csv(test_data_path)
    # Build vocabulary
    vocab = Vocabulary(SPECIAL_TOKENS)
    vocab.build_vocab(train_df)

    # Create datasets
    train_data = MazeDataset(train_df, vocab)
    val_data = MazeDataset(val_df, vocab)
    test_data = MazeDataset(test_df, vocab)

    collate = lambda b: collate_fn(b, vocab.PAD_IDX)
    train_iterator = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    val_iterator = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    test_iterator = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    # Initialize model
    model = Seq2SeqTransformer(
        input_dim=vocab.n_tokens,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_idx=vocab.PAD_IDX,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_iterator),
        epochs=EPOCHS,
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.PAD_IDX, label_smoothing=LABEL_SMOOTHING
    )

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    best_val_acc = -1
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, scheduler, criterion, CLIP)
        val_loss, val_acc = evaluate(
            model, val_iterator, criterion, vocab, use_beam_search=False
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1:02}/{EPOCHS} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": {
                        "token_to_idx": vocab.token_to_idx,
                        "idx_to_token": vocab.idx_to_token,
                    },
                    "hyperparameters": {
                        "D_MODEL": D_MODEL,
                        "NHEAD": NHEAD,
                        "NUM_LAYERS": NUM_LAYERS,
                        "DIM_FEEDFORWARD": DIM_FEEDFORWARD,
                        "DROPOUT": DROPOUT,
                    },
                    "training_info": {"best_val_acc": best_val_acc, "epoch": epoch + 1},
                },
                save_path,
            )
            print(f"  --> Model saved with Val Acc: {val_acc:.4f}")

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"{'='*80}\n")

    # Save training history
    save_prefix = save_path.replace(".pth", "")
    save_training_history(train_losses, val_losses, val_accs, save_prefix)

    # Plot training metrics
    plot_metrics(train_losses, val_losses, val_accs, save_prefix)

    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Best model loaded successfully!")

    # Test evaluation with beam search on full test set
    print("\nEvaluating on test set with beam search (this may take a while)...")
    test_loss, test_acc_full = evaluate(
        model, test_iterator, criterion, vocab, use_beam_search=True
    )
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc (Full, Beam Search): {test_acc_full:.4f}"
    )

    # Detailed test evaluation with sample selection
    test_acc, random_samples = evaluate_and_sample_test(
        model, test_data, vocab, device, num_samples=10, save_prefix=save_prefix
    )

    # Print sample predictions
    print_sample_predictions(random_samples)

    # Save samples for visualization
    save_samples_for_visualization(random_samples, save_prefix)

    # Plot test accuracy comparison
    plot_test_accuracy(test_acc, best_val_acc, save_prefix)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test Accuracy:            {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Difference (Val - Test):  {(best_val_acc - test_acc):.4f}")
    print("=" * 80)
    print(f"\nAll results saved with prefix: {save_prefix}")
    print("Files generated:")
    print(f"  - {save_path} (trained model)")
    print(f"  - {save_prefix}_training_metrics.png")
    print(f"  - {save_prefix}_test_accuracy.png")
    print(f"  - {save_prefix}_training_history.json")
    print(f"  - {save_prefix}_test_samples.json")
    print(f"  - {save_prefix}_visualization_samples/ (directory)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    TRAIN_PATH = "/kaggle/input/maze-dataset/train_6x6_mazes.csv"
    TEST_PATH = "/kaggle/input/maze-dataset/test_6x6_mazes.csv"
    SAVE_PATH = "/kaggle/working/transformer_model.pth"

    main(TRAIN_PATH, TEST_PATH, SAVE_PATH)
