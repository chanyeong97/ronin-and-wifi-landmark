import argparse
import pickle
import tensorflow as tf

from os import path as osp

from src.modules.wifi import Wifi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-ml', type=str, choices=['autoencoder', 'landmark', 'wifi_net', 'enhanced_wifi'], \
        default='autoencoder', help='select a model to train')
    parser.add_argument('--autoencoder_model', '-a', type=str, default='trained_models/autoencoder/autoencoder_model', \
        help='autoencoder model path')
    parser.add_argument('--landmark_model', '-l', type=str, default='trained_models/landmark/landmark_model', \
        help='landmark model path')
    parser.add_argument('--wifi_net_model', '-w', type=str, default='trained_models/wifi_net/wifi_net_model', \
        help='wifi_net model path')
    parser.add_argument('--enhanced_wifi_model', '-e', type=str, default='trained_models/enhanced_wifi/enhanced_wifi_model', \
        help='enhanced wifi model path')
    parser.add_argument('--train_dataset', '-trd', type=str, default='dataset/landmark_train', \
        help='train dataset path')
    parser.add_argument('--test_dataset', '-ted', type=str, \
        help='test dataset path')
    parser.add_argument('--validation_dataset', '-vad', type=str, \
        help='validation dataset path')
    parser.add_argument('--bssid', '-b', type=str, default='trained_models/bssid.pickle', \
        help='bssid path')
    parser.add_argument('--mode', '-m', type=str, choices=['train', 'test'], default='train', \
        help='train or test')
    return parser.parse_args()


def train(args):
    if args.validation_dataset:
        wifi = Wifi(train=args.train_dataset, validation=args.validation_dataset)
    else:
        wifi = Wifi(train=args.train_dataset)

    if osp.exists(args.bssid):
        wifi.load_bssid(args.bssid)
    else:
        wifi.select_bssid()
        with open(args.bssid, 'wb') as fw:
            pickle.dump(wifi.get_bssid(), fw)

    if args.model == 'autoencoder':
        print('.')
    elif args.model == 'landmark':
        print('.')
    elif args.model == 'wifi_net':
        print('.')
    elif args.model == 'enhanced_wifi':
        print('.')


def test(args):
    print('test')
    

if __name__ == '__main__':
    # print(tf.test.is_gpu_available())
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
