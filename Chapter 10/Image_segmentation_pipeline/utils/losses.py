# -*- coding: utf-8 -*-
"""Custom loss class, losses.py"""
import tensorflow as tf
from tensorflow.keras import backend as K


class CustomLossAndMetrics:
    def __init__(self):
        """
        CustomLossAndMetrics class for defining custom loss and metrics functions.
        """
        pass


    @staticmethod
    def dice_coeff(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the Dice coefficient between the true and predicted segmentation masks.


        Args:
            y_true (tf.Tensor): True segmentation masks.
            y_pred (tf.Tensor): Predicted segmentation masks.


        Returns:
            tf.Tensor: Dice coefficient score.
        """
        smooth = 1.


        # Flatten
        y_true_f = tf.reshape(tf.one_hot(tf.cast(y_true, tf.int32), 3), [-1, 3])
        y_pred_f = tf.reshape(y_pred, [-1, 3])


        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score


    @staticmethod
    def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the Dice loss between the true and predicted segmentation masks.


        Args:
            y_true (tf.Tensor): True segmentation masks.
            y_pred (tf.Tensor): Predicted segmentation masks.


        Returns:
            tf.Tensor: Dice loss.
        """
        loss = 1 - CustomLossAndMetrics.dice_coeff(y_true, y_pred)
        return loss


    @staticmethod
    def entropy_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the entropy loss between the true and predicted segmentation masks.


        Args:
            y_true (tf.Tensor): True segmentation masks.
            y_pred (tf.Tensor): Predicted segmentation masks.


        Returns:
            tf.Tensor: Entropy loss.
        """
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return loss


    @staticmethod
    def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the combined loss (entropy loss + dice loss) between the true and predicted segmentation masks.


        Args:
            y_true (tf.Tensor): True segmentation masks.
            y_pred (tf.Tensor): Predicted segmentation masks.


        Returns:
            tf.Tensor: Combined loss.
        """
        loss = CustomLossAndMetrics.entropy_loss(y_true, y_pred) + CustomLossAndMetrics.dice_loss(y_true, y_pred)
        return loss