import sys
import argparse
import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import numpy as np
import ast
from maze_rnn import (
    EncoderRNN,
    DecoderRNN,
    Attention,
    Seq2Seq_RNN,
    predict_path,
    Vocabulary as VocabRNN,
)
from maze_transformer import (
    Seq2SeqTransformer,
    PositionalEncoding,
    Vocabulary as VocabTransformer,
    beam_search_decode,
)
from visualize import plot_maze


def load_input_from_file(filepath):
    """Load and parse input sequence from file"""
    with open(filepath, "r") as f:
        content = f.read().strip()

    try:
        # Try to evaluate as Python list
        tokens = ast.literal_eval(content)
        if isinstance(tokens, list):
            return tokens
    except:
        pass

    # Try parsing as space-separated tokens
    tokens = content.split()
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Evaluate maze pathfinding model")
    parser.add_argument(
        "model_path", type=str, help="Path to trained model (.pth file)"
    )
    parser.add_argument("input_file", type=str, help="Path to input maze file (.txt)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")
    print(f"Loading input from: {args.input_file}")

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)

    # Detect model type based on hyperparameters
    if "HIDDEN_DIM" in checkpoint["hyperparameters"]:
        # ==================== RNN MODEL ====================
        print("Detected RNN model")
        model_type = "rnn"

        # Load RNN vocabulary
        vocab = VocabRNN()
        vocab.itos = checkpoint["vocab"]["itos"]
        vocab.stoi = checkpoint["vocab"]["stoi"]
        vocab.vocab_size = len(vocab.itos)
        print(f"Vocabulary size: {vocab.vocab_size}")

        hparams = checkpoint["hyperparameters"]
        INPUT_DIM = vocab.vocab_size
        OUTPUT_DIM = vocab.vocab_size

        # Build RNN model
        enc = EncoderRNN(
            INPUT_DIM,
            hparams["EMBEDDING_DIM"],
            hparams["HIDDEN_DIM"],
            hparams["NUM_RNN_LAYERS"],
            0.5,
        )
        attn = Attention(hparams["HIDDEN_DIM"])
        dec = DecoderRNN(
            OUTPUT_DIM,
            hparams["EMBEDDING_DIM"],
            hparams["HIDDEN_DIM"],
            hparams["NUM_RNN_LAYERS"],
            0.5,
            attn,
        )
        model = Seq2Seq_RNN(enc, dec, device, pad_idx=vocab.stoi.get("<pad>", 0))
        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        # ==================== TRANSFORMER MODEL ====================
        print("Detected Transformer model")
        model_type = "transformer"

        # Load Transformer vocabulary
        vocab_data = checkpoint["vocab"]
        vocab = VocabTransformer(["<PAD>", "<SOS>", "<EOS>", "<UNK>"])
        vocab.token_to_idx = vocab_data["token_to_idx"]
        vocab.idx_to_token = {int(k): v for k, v in vocab_data["idx_to_token"].items()}
        vocab.n_tokens = len(vocab.token_to_idx)
        vocab.PAD_IDX = vocab.token_to_idx["<PAD>"]
        vocab.SOS_IDX = vocab.token_to_idx["<SOS>"]
        vocab.EOS_IDX = vocab.token_to_idx["<EOS>"]
        vocab.UNK_IDX = vocab.token_to_idx["<UNK>"]
        print(f"Vocabulary size: {vocab.n_tokens}")

        hparams = checkpoint["hyperparameters"]

        # Build Transformer model
        model = Seq2SeqTransformer(
            input_dim=vocab.n_tokens,
            d_model=hparams["D_MODEL"],
            nhead=hparams["NHEAD"],
            num_layers=hparams["NUM_LAYERS"],
            dim_feedforward=hparams["DIM_FEEDFORWARD"],
            dropout=hparams["DROPOUT"],
            pad_idx=vocab.PAD_IDX,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

    print("Model loaded successfully!")

    # Load input
    input_tokens = load_input_from_file(args.input_file)
    print(f"\nInput sequence length: {len(input_tokens)}")
    print(f"First 10 tokens: {input_tokens[:10]}")

    # Predict
    print("\nGenerating prediction...")
    if model_type == "rnn":
        # Use the predict_path function from maze_rnn.py
        predicted_path = predict_path(model, input_tokens, vocab, device, max_len=100)
    else:
        # Use beam_search_decode from maze_transformer.py
        # Numericalize input
        numerical = vocab.numericalize(input_tokens, is_target=False)
        src = (
            torch.LongTensor(numerical).unsqueeze(-1).to(device)
        )  # Shape: (seq_len, 1)

        # Generate prediction
        predicted_path = beam_search_decode(
            model, src, vocab, beam_width=5, max_len=100
        )

    print("\n" + "=" * 80)
    print("PREDICTED PATH:")
    print("=" * 80)
    print(predicted_path)
    print("=" * 80)

    # Visualize
    print("\nVisualizing maze with predicted path...")
    # Construct the full sequence for visualization
    viz_tokens = input_tokens + ["<PATH_START>"] + predicted_path + ["<PATH_END>"]
    plot_maze(viz_tokens)

    print(f"\nPrediction complete using {model_type.upper()} model!")


if __name__ == "__main__":
    main()
