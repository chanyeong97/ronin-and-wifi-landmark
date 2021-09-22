import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from os import path as osp

from config import *
from src.modules.train import model_fit
from src.modules.loss import autoencoder_loss


class Autoencoder(Model):
    def __init__(self, x):
        super(Autoencoder, self).__init__()
        self.dense1 = Dense(ENCODING_LAYER_1, activation='relu')
        self.dense2 = Dense(ENCODING_LAYER_2, activation='relu')
        self.dense3 = Dense(ENCODING_LAYER_3, activation='relu')
        self.dense4 = Dense(DECODING_LAYER_1, activation='relu')
        self.dense5 = Dense(DECODING_LAYER_2, activation='relu')
        self.dense6 = Dense(len(x))
        self.dropout = Dropout(AUTOENCODER_DROPOUT)

    def __call__(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        x = self.dropout(x, training=training)
        x = self.dense4(x)
        x = self.dropout(x, training=training)
        x = self.dense5(x)
        x = self.dropout(x, training=training)
        x = self.dense6(x)
        return x

    def encoder(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


def set_autoencoder_input(data):
    x = []
    for sequence in data:
        for i in range(len(data[sequence])):
            x.append(data[sequence][i]['x'])
    return np.array(x)


def train_autoencoder(args, wifi):
    wifi.sort_by_bssid(wifi.get_train_data())
    wifi.sort_by_bssid(wifi.get_test_data())
    train_data = wifi.get_train_data()
    test_data = wifi.get_test_data()

    train_x = set_autoencoder_input(train_data)
    test_x = set_autoencoder_input(test_data)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_x))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_x))
    train_dataset = train_dataset.batch(batch_size=AUTOENCODER_BATCH_SIZE)

    model = Autoencoder(wifi.get_bssid())
    optimizer = tf.optimizers.Adam(learning_rate=AUTOENCODER_LEARNING_RATE)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, args.autoencoder, max_to_keep=AUTOENCODER_MAX_TO_KEEP)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    best_test_loss = autoencoder_loss(model(test_x), test_x)

    for epoch in range(AUTOENCODER_EPOCHS):
        loss = []
        for x, y in train_dataset:
            loss.append(model_fit(x, y, autoencoder_loss, model, optimizer))
        loss_avg = sum(loss) / len(loss)
        ckpt.step.assign_add(1)
        if (epoch+1) % 250 == 0:
            train_loss = autoencoder_loss(model(train_x), train_x)
            test_loss = autoencoder_loss(model(test_x), test_x)
            print("epoch: ", (epoch+1))
            print("train loss: ", loss_avg)
            print("train loss with dropout: ", train_loss)
            print("test loss without dropout: ", test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                model.save_weights(osp.join(args.autoencoder, 'model', 'model'))


def test_autoencoder(args, wifi):
    wifi.sort_by_bssid(wifi.get_test_data())
    test_data = wifi.get_test_data()
    test_x = set_autoencoder_input(test_data)

    model = Autoencoder(wifi.get_bssid())
    model.load_weights(osp.join(args.autoencoder, 'model', 'model'))
    pred = model(test_x)
    loss = autoencoder_loss(pred, test_x)
    print("loss: ", loss)