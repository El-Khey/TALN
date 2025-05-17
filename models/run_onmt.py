"""This example demonstrates how to train a standard Transformer model in a few
lines of code using OpenNMT-tf high-level APIs.
"""

import argparse
import logging
import tensorflow as tf
import opennmt as onmt


tf.get_logger().setLevel(logging.INFO)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run", choices=["train", "translate"], help="Action à exécuter."
    )
    args = parser.parse_args()

    config = {
        "model_dir": "model-checkpoints/",
        "train": {
            "batch_size": 64,
            "max_step": 100,
            "effective_batch_size": 1024,  # réduit la lenteur liée à l'accumulation
        },
        "data": {
            "source_vocabulary": "data/src-vocab.txt",
            "target_vocabulary": "data/tgt-vocab.txt",
            "train_features_file": "data/train.en",
            "train_labels_file": "data/train.fr",
        },
        "model": {
            "num_layers": 2,
            "num_units": 128,
            "num_heads": 4,
            "ffn_inner_dim": 256,
            "dropout": 0.1,
        },
    }

    model = onmt.models.TransformerBase()
    runner = onmt.Runner(model, config, auto_config=True)

    if args.run == "train":
        runner.train()
    elif args.run == "translate":
        runner.infer("data/test.en")


if __name__ == "__main__":
    main()
