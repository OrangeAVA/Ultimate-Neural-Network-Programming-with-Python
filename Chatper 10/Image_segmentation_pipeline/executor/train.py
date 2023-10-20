import random
import os
import json
import tensorflow as tf


# Internal imports
from model.att_unet import Attention_UNet
from custom_dataloader.dataloader import OxfordPetsDataLoader
from utils.losses import CustomLossAndMetrics


class AttentionUNetTrainer:
    def __init__(self, input_dir: str, target_dir: str, config_path: str = "D:/mine/AISummer/project2/configs/config.json"): # Give path to you config file
        """
        AttentionUNetTrainer class to train the Attention UNet model on the Oxford Pets dataset.


        Args:
            input_dir (str): Directory containing input images.
            target_dir (str): Directory containing target images.
            config_path (str, optional): Path to the configuration file. Defaults to "D:/mine/AISummer/project2/configs/config.json".
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.config_file = config_path


        # Load the configuration file
        with open(self.config_file, "r") as f:
            config = json.load(f)


        self.img_size = tuple(config["data"]["image_size"])
        self.batch_size = config["train"]["batch_size"]
        self.val_samples = config["train"]["val_samples"]
        self.epochs = config["train"]["epochs"]


        self.input_img_paths = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".jpg")
            ]
        )
        self.target_img_paths = sorted(
            [
                os.path.join(target_dir, fname)
                for fname in os.listdir(target_dir)
                if fname.endswith(".png") and not fname.startswith(".")
            ]
        )


        random.Random(1337).shuffle(self.input_img_paths)
        random.Random(1337).shuffle(self.target_img_paths)
        self.train_input_img_paths = self.input_img_paths[:-self.val_samples]
        self.train_target_img_paths = self.target_img_paths[:-self.val_samples]
        self.val_input_img_paths = self.input_img_paths[-self.val_samples:]
        self.val_target_img_paths = self.target_img_paths[-self.val_samples:]
        self.val_gen = OxfordPetsDataLoader(self.batch_size, self.img_size, self.val_input_img_paths, self.val_target_img_paths)


    def train(self):
        """
        Trains the Attention UNet model on the Oxford Pets dataset.
        """
        model = Attention_UNet().build()
        model.compile(
            optimizer="adam", 
            loss=CustomLossAndMetrics.combined_loss, 
            metrics=[CustomLossAndMetrics.dice_coeff]
        )


        train_gen = OxfordPetsDataLoader(self.batch_size, self.img_size, self.train_input_img_paths, self.train_target_img_paths)
        val_gen = OxfordPetsDataLoader(self.batch_size, self.img_size, self.val_input_img_paths, self.val_target_img_paths)


        callbacks = [
            tf.keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
        ]


        # Prefetch the next 2 batches for training and validation data loaders
        train_gen.prefetch(2)
        val_gen.prefetch(2)


        model.fit(train_gen, epochs=self.epochs, validation_data=val_gen, callbacks=callbacks)
