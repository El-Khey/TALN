"""This example demonstrates how to train a standard Transformer model in a few
lines of code using OpenNMT-tf high-level APIs. It includes functionality for training,
translation, and a CSV export for comparing translated output.
"""

import argparse
import logging
import os
import tensorflow as tf
import opennmt as onmt
import numpy as np
import pandas as pd
import csv

print("GPUs Available: ", tf.config.list_physical_devices("GPU"))

# print pandas and numpy versions
print("Using OpenNMT-tf version:", onmt.__version__)
print("Using TensorFlow version:", tf.__version__)
print("Using pandas version:", pd.__version__)
print("Using numpy version:", np.__version__)


tf.get_logger().setLevel(logging.INFO)


def export_comparison_csv(source_file, target_file, predictions_file, output_csv):
    """
    Export a CSV file with original, expected, and translated text.
    """
    with open(source_file, "r", encoding="utf-8") as src_file, open(
        target_file, "r", encoding="utf-8"
    ) as tgt_file, open(predictions_file, "r", encoding="utf-8") as pred_file, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as out_csv:

        source_lines = src_file.readlines()
        target_lines = tgt_file.readlines()
        pred_lines = pred_file.readlines()

        writer = csv.writer(out_csv)
        writer.writerow(["Source (FR)", "Target (EN)", "Translation (EN)"])

        for src, tgt, pred in zip(source_lines, target_lines, pred_lines):
            writer.writerow([src.strip(), tgt.strip(), pred.strip()])

    print(f"Comparison CSV exported to {output_csv}")


def main():
    # Define constants
    PREDICTIONS_FILE = "output.txt"
    CSV_OUTPUT_FILE = "translation_comparison.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run",
        choices=["train", "translate", "export-csv", "export"],
        help="Action à exécuter (train, translate, export-csv, export).",
    )
    parser.add_argument(
        "--steps", type=int, default=5000, help="Number of training steps"
    )
    args = parser.parse_args()

    # Define data paths
    data_dir = "../data/cleaned"
    source_vocab = os.path.join(data_dir, "vocab.fr")
    target_vocab = os.path.join(data_dir, "vocab.en")
    train_source = os.path.join(data_dir, "train.fr")
    train_target = os.path.join(data_dir, "train.en")
    test_source = os.path.join(data_dir, "test.fr")
    test_target = os.path.join(data_dir, "test.en")

    config = {
        "model_dir": "model-checkpoints/",
        "train": {
            "batch_size": 4096,
            "batch_type": "tokens",
            "effective_batch_size": 8192,
            "max_step": args.steps,
            "save_checkpoints_steps": 500,
            "keep_checkpoint_max": 5,
            "average_last_checkpoints": 5,
            "mixed_precision": True,
        },
        "data": {
            "source_vocabulary": source_vocab,
            "target_vocabulary": target_vocab,
            "train_features_file": train_source,
            "train_labels_file": train_target,
            "eval_features_file": test_source,
            "eval_labels_file": test_target,
            "num_threads": 4,
        },
        "params": {
            "device_placement": "gpu",
            "optimizer": "LazyAdam",
            "learning_rate": 2.0,  # Initial learning rate
            "decay_type": "NoamDecay",  # Learning rate decay - correct casing
            "decay_params": {
                "model_dim": 512,  # Model dimension
                "warmup_steps": 4000,  # Warmup steps for learning rate
            },
        },
    }

    # Create the TransformerBase model
    model = onmt.models.TransformerBase()

    runner = onmt.Runner(model, config, auto_config=True)

    if args.run == "train":
        runner.train()
    elif args.run == "export":
        runner.export(export_dir="exported_model", checkpoint_path="model-checkpoints/")
        print("Model exported successfully.")
    elif args.run == "translate":
        runner.infer(test_source, predictions_file=PREDICTIONS_FILE)
        print(f"Translation completed. Results saved to {PREDICTIONS_FILE}")
    elif args.run == "export-csv":
        export_comparison_csv(
            test_source, test_target, PREDICTIONS_FILE, CSV_OUTPUT_FILE
        )


if __name__ == "__main__":
    main()
