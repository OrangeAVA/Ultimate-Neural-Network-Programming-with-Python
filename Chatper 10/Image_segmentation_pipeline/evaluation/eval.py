# -*- coding: utf-8 -*-
"""evaluation class, eval.py"""


from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import ImageOps
from utils.losses import CustomLossAndMetrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class Evaluator:
    def __init__(self, trainer):
        """
        Evaluator class to evaluate the model's predictions on the validation dataset.


        Args:
            trainer: Instance of the AttentionUNetTrainer class.
        """
        self.trainer = trainer
        self.model = tf.keras.models.load_model('oxford_segmentation.h5', 
                                                custom_objects={'combined_loss': CustomLossAndMetrics.combined_loss, 
                                                                'dice_coeff': CustomLossAndMetrics.dice_coeff})


    def evaluate(self):
        """
        Evaluate the model's predictions on the validation dataset and display the results.
        """
        val_gen = self.trainer.val_gen
        val_preds = self.model.predict(val_gen)
        
        def display_mask(mask: np.ndarray) -> np.ndarray:
            """
            Quick utility to display a model's prediction.


            Args:
                mask (np.ndarray): Predicted segmentation mask.


            Returns:
                np.ndarray: RGB mask for visualization.
            """
            color_map = {
                0: np.array([0, 0, 0]),
                1: np.array([0, 120, 120]),
                2: np.array([120, 0, 120]),
            }
            rgb_mask = np.zeros((*mask.shape, 3))
            for k in color_map.keys():
                rgb_mask[mask == k] = color_map[k]
            return rgb_mask


        # Display results for validation image #5
        i = 5


        # Load and display input image
        input_img = img_to_array(load_img(self.trainer.val_input_img_paths[i], target_size=self.trainer.img_size))
        predicted_mask = display_mask(np.argmax(val_preds[i], axis=-1))
        target_mask = display_mask(np.squeeze(img_to_array(load_img(self.trainer.val_target_img_paths[i], target_size=self.trainer.img_size, color_mode="grayscale")).astype(int)))


        fig, ax = plt.subplots(1, 3, figsize=(20, 20))
        ax[0].imshow(input_img/255.0)
        ax[0].title.set_text('Input Image')
        ax[0].axis('off')


        ax[1].imshow(predicted_mask/255.0)
        ax[1].title.set_text('Predicted Mask')
        ax[1].axis('off')


        ax[2].imshow(target_mask/255.0)
        ax[2].title.set_text('Ground Truth Mask')
        ax[2].axis('off')
        plt.show()
