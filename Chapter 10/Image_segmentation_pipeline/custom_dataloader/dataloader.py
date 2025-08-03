# -*- coding: utf-8 -*-
"""Data Loader, custom_dataloader.py"""


import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import random



class OxfordPetsDataLoader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        """
        OxfordPetsDataLoader class to load and preprocess image data for the Oxford Pets dataset.


        Args:
            batch_size (int): Number of samples per batch.
            img_size (tuple): Tuple representing the target image size in the format (height, width).
            input_img_paths (list): List of input image paths.
            target_img_paths (list): List of target image paths.
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths


    def __len__(self):
        """
        Returns the number of batches in the Sequence.


        Returns:
            int: Number of batches.
        """
        return len(self.target_img_paths) // self.batch_size


    def __getitem__(self, idx):
        """
        Generates one batch of data.


        Args:
            idx (int): Index of the batch.


        Returns:
            tuple: A tuple containing the input and target image batches.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]


        # Preallocate arrays for input and target images
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype=np.float32)
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype=np.uint8)


        # Load and preprocess input images
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img_to_array(img)


        # Load and preprocess target images
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img_to_array(img)[:, :, 0], axis=2)
            y[j] -= 1


        return x, y


    def on_epoch_end(self):
        """
        Method called at the end of each epoch.
        """
        # Shuffle the input and target image paths
        random.Random(1337).shuffle(self.input_img_paths)
        random.Random(1337).shuffle(self.target_img_paths)


    def prefetch(self, num_prefetch):
        """
        Prefetches the next 'num_prefetch' batches of data.


        Args:
            num_prefetch (int): Number of batches to prefetch.
        """
        for _ in range(num_prefetch):
            self.__getitem__(np.random.randint(0, self.__len__()))
