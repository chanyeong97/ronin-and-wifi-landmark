import tensorflow as tf
import numpy as np


def autoencoder_loss(pred, y):
    return tf.reduce_mean(tf.square(pred - y))