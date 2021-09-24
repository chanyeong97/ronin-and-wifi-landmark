import argparse
import pickle
import tensorflow as tf
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from os import path as osp
from glob import glob

from config import *
from src.modules.wifi import Wifi
from src.modules.dataset import ronin_features
from src.models.autoencoder import Autoencoder, train_autoencoder, test_autoencoder
from src.models.landmark import Landmark, train_landmark, test_landmark
from src.models.ronin import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-ml', type=str, \
        choices=['autoencoder', 'landmark', 'test'], \
        default='test', help='select a model to train')
    parser.add_argument('--autoencoder', '-a', type=str, default='trained_models/autoencoder', \
        help='autoencoder model path')
    parser.add_argument('--landmark', '-l', type=str, default='trained_models/landmark', \
        help='landmark model path')
    parser.add_argument('--ronin', '-r', type=str, default='trained_models/ronin_resnet/checkpoint_gsn_latest.pt', \
        help='ronin resnet model path')

    parser.add_argument('--train_dataset', '-trd', type=str, #default='dataset/landmark', \
        help='train dataset path')
    parser.add_argument('--test_dataset', '-ted', type=str, default='dataset/test', \
        help='test dataset path')
    parser.add_argument('--bssid', '-b', type=str, default='trained_models/bssid.pickle', \
        help='bssid path')
    parser.add_argument('--mode', '-m', type=str, choices=['train', 'test'], default='test', \
        help='train or test')
    return parser.parse_args()


def set_test_wifi(data):
    x = []
    t = []
    for i in range(len(data)):
        x.append(data[i]['x'])
        t.append(data[i]['timestamp'])
    return np.array(x), t


def rotate(velocity, theta):
    x = velocity[0][0].copy()
    y = velocity[0][1].copy()
    velocity[0][0] = np.cos(theta)*x - np.sin(theta)*y
    velocity[0][1] = np.sin(theta)*x + np.cos(theta)*y
    return velocity


def localization(pre_landmark, cur_landmark, position, alpha, theta, pre_position):
    if pre_landmark == cur_landmark or cur_landmark == 0:
        return pre_landmark, position, alpha, theta, pre_position

    if pre_landmark == 0 and alpha == 1:
        position = LANDMARK_POSITION[str(cur_landmark)]
        pre_position = position.copy()
    else:
        cur_position = LANDMARK_POSITION[str(pre_landmark)+'to'+str(cur_landmark)]
        real_vector = cur_position - pre_position
        real_vector_absolute_value = np.linalg.norm(real_vector)
        pred_vector = position - pre_position
        pred_vector_absolute_value = np.linalg.norm(pred_vector)
        if alpha == 1:
            alpha *= (real_vector_absolute_value/pred_vector_absolute_value)
        else:
            a = alpha*(real_vector_absolute_value/pred_vector_absolute_value)
            alpha = alpha + (a-alpha)*0.8

        if np.cross(pred_vector[0], real_vector[0]) > 0:
            theta += (np.arccos(np.dot(real_vector[0], pred_vector[0]) / (real_vector_absolute_value*pred_vector_absolute_value)))
        else:
            theta -= (np.arccos(np.dot(real_vector[0], pred_vector[0]) / (real_vector_absolute_value*pred_vector_absolute_value)))
        position = cur_position.copy()
        pre_position = cur_position.copy()
    
    pre_landmark = cur_landmark.copy()
    return pre_landmark, position, alpha, theta, pre_position


def train(args, wifi):
    if args.model == 'autoencoder':
        train_autoencoder(args, wifi)
    elif args.model == 'landmark':
        train_landmark(args, wifi)


def test(args, wifi):
    if args.model == 'autoencoder':
        test_autoencoder(args, wifi)
    elif args.model == 'landmark':
        test_landmark(args, wifi)
    elif args.model == 'test':
        if not torch.cuda.is_available():
            device = torch.device('cpu')
            checkpoint = torch.load(args.ronin, map_location=lambda storage, location: storage)
        else:
            device = torch.device('cuda:0')
            checkpoint = torch.load(args.ronin)

        ronin = get_model()

        ronin.load_state_dict(checkpoint['model_state_dict'])
        ronin.eval().to(device)
        print('Model {} loaded to device {}.'.format(args.ronin, device))

        autoencoder = Autoencoder(wifi.get_bssid())
        autoencoder.load_weights(osp.join(args.autoencoder, 'model', 'model'))

        landmark = Landmark()
        landmark.load_weights(osp.join(args.landmark, 'model', 'model'))

        wifi.sort_by_bssid(wifi.get_test_data())
        data = wifi.get_test_data()
        all_sequence = glob(osp.join(args.test_dataset, '*'))
        for sequence in all_sequence:
            position = np.array([[0, 0]], dtype=np.float32)
            trajectory = [position[0].copy()]
            pre_landmark = np.array(0)
            pre_position = position.copy()
            alpha = 1
            theta = 0
            ronin_input = []
            _, features = ronin_features(sequence)
            wifi_x, wifi_t = set_test_wifi(data[osp.basename(sequence)])
            wifi_x = autoencoder.encoder(wifi_x).numpy()
            for i in range(len(features['timestamp'])):
                cur_timestamp = features['timestamp'][i]
                cur_feature = features['feature'][i]
                ronin_input.append(cur_feature)
                if len(ronin_input) == 200:
                    ronin_x = torch.tensor([np.array(ronin_input, dtype=np.float32).T])
                    velocity = ronin(ronin_x.to(device)).cpu().detach().numpy() * alpha
                    velocity = rotate(velocity, theta)
                    position += velocity*(RONIN_INTERVAL*RONIN_STEP_SIZE)
                    trajectory.append(position[0].copy())
                    del ronin_input[:10]

                if len(wifi_t):
                    if wifi_t[0] < cur_timestamp:
                        cur_landmark = np.argmax(landmark(wifi_x[0].reshape([1, -1])).numpy(), axis=-1)[0]
                        pre_landmark, position, alpha, theta, pre_position = localization(
                            pre_landmark, 
                            cur_landmark, 
                            position, 
                            alpha, 
                            theta, 
                            pre_position)

                        del wifi_t[0]
                        wifi_x = np.delete(wifi_x, (0), axis=0)
            
            trajectory = np.array(trajectory)
            x = trajectory[50:, 0]
            y = trajectory[50:, 1]
            plt.plot(x, y, '.--')
            plt.show()
            print('.')
    

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    args = parse_args()

    wifi = Wifi(train=args.train_dataset, test=args.test_dataset)

    if osp.exists(args.bssid):
        wifi.load_bssid(args.bssid)
    else:
        wifi.select_bssid()
        with open(args.bssid, 'wb') as fw:
            pickle.dump(wifi.get_bssid(), fw)

    if args.mode == 'train':
        train(args, wifi)
    elif args.mode == 'test':
        test(args, wifi)
