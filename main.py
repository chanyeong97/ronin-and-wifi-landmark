import argparse
import pickle
import tensorflow as tf

from os import path as osp

from src.modules.wifi import Wifi
from src.models.autoencoder import train_autoencoder, test_autoencoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-ml', type=str, choices=['autoencoder', 'landmark', 'wifi_net', 'enhanced_wifi'], \
        default='autoencoder', help='select a model to train')
    parser.add_argument('--autoencoder', '-a', type=str, default='trained_models/autoencoder', \
        help='autoencoder model path')
    parser.add_argument('--landmark', '-l', type=str, default='trained_models/landmark', \
        help='landmark model path')
    parser.add_argument('--wifi_net', '-w', type=str, default='trained_models/wifi_net', \
        help='wifi_net model path')
    parser.add_argument('--enhanced_wifi', '-e', type=str, default='trained_models/enhanced_wifi', \
        help='enhanced wifi model path')
    parser.add_argument('--train_dataset', '-trd', type=str, #default='dataset/autoencoder_train', \
        help='train dataset path')
    parser.add_argument('--test_dataset', '-ted', type=str, default='dataset/landmark_train', \
        help='test dataset path')
    parser.add_argument('--bssid', '-b', type=str, default='trained_models/bssid.pickle', \
        help='bssid path')
    parser.add_argument('--mode', '-m', type=str, choices=['train', 'test'], default='test', \
        help='train or test')
    return parser.parse_args()


def train(args, wifi):
    if args.model == 'autoencoder':
        train_autoencoder(args, wifi)
    elif args.model == 'landmark':
        print('.')
    elif args.model == 'wifi_net':
        print('.')
    elif args.model == 'enhanced_wifi':
        print('.')


def test(args, wifi):
    if args.model == 'autoencoder':
        test_autoencoder(args, wifi)
    elif args.model == 'landmark':
        print('.')
    elif args.model == 'wifi_net':
        print('.')
    elif args.model == 'enhanced_wifi':
        print('.')
    

if __name__ == '__main__':
    print(tf.test.is_gpu_available())
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
