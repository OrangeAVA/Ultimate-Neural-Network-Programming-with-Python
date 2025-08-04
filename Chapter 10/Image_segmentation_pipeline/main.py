# -*- coding: utf-8 -*-
"""main.py"""


import json
import tensorflow as tf
from executor.train import AttentionUNetTrainer
from evaluation.eval import Evaluator


def main():
    # Change these paths and set your directories here, also change paths in other files as well
    input_dir = "images"
    target_dir = "annotations/trimaps"
    config_file = "configs/config.json"


    trainer = AttentionUNetTrainer(input_dir, target_dir, config_file)
    trainer.train()


    evaluator = Evaluator(trainer)  # Initialize the evaluator with the trainer
    evaluator.evaluate()  # Evaluate the model


if __name__ == "__main__":
    main()
