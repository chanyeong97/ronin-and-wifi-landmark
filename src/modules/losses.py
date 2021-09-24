import tensorflow as tf
import numpy as np


def autoencoder_loss(pred, y):
    return tf.reduce_mean(tf.square(pred - y))


def landmark_loss(pred, y):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, pred, from_logits=True))
