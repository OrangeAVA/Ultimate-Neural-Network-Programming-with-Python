# -*- coding: utf-8 -*-
"""Unet model with attention gates"""


import json
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Input, MaxPool2D, Conv2DTranspose, concatenate, Activation
from tensorflow.keras import backend as K



class CNNBlocks:
    """
    A class used to create Convolution Blocks of UNet Model
    """


    def __init__(self, kernel_size: int):
        """
        Initializes the CNNBlocks class with kernel size and other parameters
        Args:
            kernel_size (int): The size of the convolution kernel
        """
        self.activation = "relu"
        self.reg = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
        self.kernel = kernel_size
        self.dropout = 0.1
        self.output_channel = 3


    def conv_down(self, n_conv: int, inputs: tf.Tensor) -> tf.Tensor:
        """
        Creates a down-convolution block with the given parameters
        Args:
            n_conv (int): Number of filters in the convolution layer
            inputs (tf.Tensor): Input tensor
        Returns:
            tf.Tensor: Output tensor
        """
        cd = Conv2D(n_conv, self.kernel, activation=self.activation,
                    kernel_regularizer=self.reg, padding='same')(inputs)
        cd = Dropout(self.dropout)(cd)
        cd = Conv2D(n_conv, self.kernel, activation=self.activation,
                    kernel_regularizer=self.reg, padding='same')(cd)


        return cd


    def attention_block(self, g: tf.Tensor, x: tf.Tensor, n_intermediate_filters: int) -> tf.Tensor:
        """
        Creates an attention block that learns to focus on specific areas of the image
        Args:
            g (tf.Tensor): Gating tensor
            x (tf.Tensor): Input tensor
            n_intermediate_filters (int): Number of filters in the intermediate convolution layer
        Returns:
            tf.Tensor: Output tensor
        """
        gating = Conv2D(n_intermediate_filters, (1, 1), padding='same')(g)  # gating mechanism
        x = Conv2D(n_intermediate_filters, (2, 2), padding='same')(x)
        attention = tf.keras.layers.Add()([gating, x])
        attention = Activation('relu')(attention)
        attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(attention)
        return tf.keras.layers.Multiply()([x, attention])


    def concat(self, n_conv: int, inputs: tf.Tensor, skip: tf.Tensor) -> tf.Tensor:
        """
        Concatenates the input and skip connection with attention
        Args:
            n_conv (int): Number of filters in the transposed convolution layer
            inputs (tf.Tensor): Input tensor
            skip (tf.Tensor): Skip connection tensor
        Returns:
            tf.Tensor: Output tensor
        """
        con = Conv2DTranspose(n_conv, (2, 2), strides=(2, 2), padding='same')(inputs)
        con = concatenate([self.attention_block(con, skip, n_conv), con])


        return con



class Attention_UNet:
    """
    Unet Model class with attention gates
    """


    def __init__(self, config_path="D:/mine/AISummer/project2/configs/config.json"): # Give path to your config file
        self.output_channels = 3
        self.config_file = config_path


        # Load the configuration file
        with open(self.config_file, "r") as f:
            config = json.load(f)


        self.img_size = tuple(config["data"]["image_size"])
        self.height = self.img_size[0]
        self.width = self.img_size[1]


    def build(self) -> tf.keras.Model:
        """
        Builds the keras model with attention gates
        Returns:
            tf.keras.Model: The built model
        """


        inputs = tf.keras.layers.Input(shape=[self.height, self.width, 3])


        conv_block = CNNBlocks(kernel_size=3)


        # Down block
        d1 = conv_block.conv_down(16, inputs)
        p1 = MaxPool2D((2, 2))(d1)
        d2 = conv_block.conv_down(32, p1)
        p2 = MaxPool2D((2, 2))(d2)
        d3 = conv_block.conv_down(64, p2)
        p3 = MaxPool2D((2, 2))(d3)
        d4 = conv_block.conv_down(128, p3)
        p4 = MaxPool2D((2, 2))(d4)
        d5 = conv_block.conv_down(256, p4)


        # Up block
        u6 = conv_block.concat(128, d5, d4)
        u6 = conv_block.conv_down(128, u6)
        u7 = conv_block.concat(64, u6, d3)
        u7 = conv_block.conv_down(64, u7)
        u8 = conv_block.concat(32, u7, d2)
        u8 = conv_block.conv_down(32, u8)
        u9 = conv_block.concat(16, u8, d1)
        u9 = conv_block.conv_down(16, u9)


        outputs = Conv2D(self.output_channels, 3, activation="softmax", padding="same")(u9)


        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])


        return self.model
