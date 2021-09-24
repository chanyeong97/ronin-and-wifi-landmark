import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from os import path as osp

from config import *
from src.models.autoencoder import Autoencoder
from src.modules.train import model_fit
from src.modules.losses import landmark_loss


class Landmark(Model):
    def __init__(self):
        super(Landmark, self).__init__()
        self.dense_1 = Dense(LANDMARK_LAYER_1, activation='relu')
        self.dense_2 = Dense(LANDMARK_LAYER_2, activation='relu')
        self.dense_3 = Dense(LANDMARK_LAYER_3, activation='relu')
        self.dense_4 = Dense(LANDMARK_LAYER_4, activation='relu')
        self.dense_5 = Dense(LANDMARK_OUTPUT_LAYER, activation='relu')
        self.dropout = Dropout(LANDMARK_DROPOUT)

    def __call__(self, x, training=False):
        x = self.dense_1(x)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)
        x = self.dropout(x, training=training)
        x = self.dense_3(x)
        x = self.dropout(x, training=training)
        x = self.dense_4(x)
        x = self.dropout(x, training=training)
        x = self.dense_5(x)
        return x


def set_landmark_features(data):
    x = []
    y = []
    for sequence in data:
        for i in range(len(data[sequence])):
            x.append(data[sequence][i]['x'])
            y.append(data[sequence][i]['landmark'])
    return np.array(x), np.array(y)


def train_landmark(args, wifi):
    wifi.sort_by_bssid(wifi.get_train_data())
    wifi.sort_by_bssid(wifi.get_test_data())
    train_data = wifi.get_train_data()
    test_data = wifi.get_test_data()

    train_x, train_y = set_landmark_features(train_data)
    test_x, test_y = set_landmark_features(test_data)

    autoencoder = Autoencoder(wifi.get_bssid())
    autoencoder.load_weights(osp.join(args.autoencoder, 'model', 'model'))

    train_x = autoencoder.encoder(train_x).numpy()
    test_x = autoencoder.encoder(test_x)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_x))
    train_dataset = train_dataset.batch(batch_size=LANDMARK_BATCH_SIZE)

    model = Landmark()
    optimizer = tf.optimizers.Adam(learning_rate=LANDMARK_LEARNING_RATE)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, args.landmark, max_to_keep=LANDMARK_MAX_TO_KEEP)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    best_test_loss = landmark_loss(model(test_x), test_y)

    for epoch in range(LANDMARK_EPOCHS):
        loss = []
        for x, y in train_dataset:
            loss.append(model_fit(x, y, landmark_loss, model, optimizer))

        loss_avg = sum(loss) / len(loss)
        ckpt.step.assign_add(1)
        if (epoch+1) % 100 == 0:
            train_loss = landmark_loss(model(train_x), train_y)
            test_loss = landmark_loss(model(test_x), test_y)
            print("epoch: ", (epoch+1))
            print("train loss: ", loss_avg)
            print("train loss with dropout: ", train_loss)
            print("test loss without dropout: ", test_loss)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                model.save_weights(osp.join(args.landmark, 'model', 'model'))

    best_model = Landmark()
    best_model.load_weights(osp.join(args.landmark, 'model', 'model'))
    pred = np.argmax(best_model(test_x).numpy(), axis=1)
    count = len(np.where(pred==test_y)[0])
    accuracy = (count/len(pred)) * 100
    print('accuracy: {}%'.format(accuracy))


def test_landmark(args, wifi):
    wifi.sort_by_bssid(wifi.get_test_data())
    test_data = wifi.get_test_data()
    test_x, test_y = set_landmark_features(test_data)

    autoencoder = Autoencoder(wifi.get_bssid())
    autoencoder.load_weights(osp.join(args.autoencoder, 'model', 'model'))

    test_x = autoencoder.encoder(test_x)

    model = Landmark()
    model.load_weights(osp.join(args.landmark, 'model', 'model'))
    pred = np.argmax(model(test_x).numpy(), axis=1)
    count = len(np.where(pred==test_y)[0])
    accuracy = (count/len(pred)) * 100
    # loss = landmark_loss(pred, test_y)
    # print("loss: ", loss)
    print('accuracy: {}%'.format(accuracy))
