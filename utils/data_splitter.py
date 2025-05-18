#!/usr/bin/env python3

import pandas as pd
import os
import subprocess
from sklearn.model_selection import train_test_split


def build_vocabulary(train_file, vocab_file, vocab_size=5000):
    """Build vocabulary using OpenNMT's build-vocab tool"""
    cmd = [
        "onmt-build-vocab",
        "--size",
        str(vocab_size),
        "--save_vocab",
        vocab_file,
        train_file,
    ]
    subprocess.run(cmd, check=True)


def split_data(csv_path, output_dir, test_size=0.2, random_state=42, vocab_size=5000):
    """
    Split the parallel corpus into train and test sets and generate vocabularies.

    Args:
        csv_path (str): Path to the CSV file containing parallel text
        output_dir (str): Directory to save the split files
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        vocab_size (int): Size of the vocabulary to generate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Split the data into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # Save train files
    train_en_path = os.path.join(output_dir, "train.en")
    train_fr_path = os.path.join(output_dir, "train.fr")
    vocab_en_path = os.path.join(output_dir, "vocab.en")
    vocab_fr_path = os.path.join(output_dir, "vocab.fr")

    train_df["en"].to_csv(train_en_path, index=False, header=False)
    train_df["fr"].to_csv(train_fr_path, index=False, header=False)

    # Save test files
    test_df["en"].to_csv(os.path.join(output_dir, "test.en"), index=False, header=False)
    test_df["fr"].to_csv(os.path.join(output_dir, "test.fr"), index=False, header=False)

    # Build vocabularies
    print("Building English vocabulary...")
    build_vocabulary(train_en_path, vocab_en_path, vocab_size)
    print("Building French vocabulary...")
    build_vocabulary(train_fr_path, vocab_fr_path, vocab_size)

    # Print statistics
    print(f"Data split complete:")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split parallel corpus into train and test sets and generate vocabularies"
    )
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for split files"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proportion of data for testing"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=5000,
        help="Size of the vocabulary to generate",
    )

    args = parser.parse_args()
    split_data(
        args.input, args.output_dir, args.test_size, args.random_seed, args.vocab_size
    )
