import argparse
import os
import pickle
import matplotlib.pyplot as plt

from glob import glob
from os import path as osp

from config import *
from src.modules.dataset import *
from src.models.ronin import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ronin', '-r', type=str, default='trained_models/ronin_resnet/checkpoint_gsn_latest.pt', \
        help='ronin resnet model path')
    parser.add_argument('--dataset', '-d', type=str, default='dataset/test', \
        help='dataset path')
    return parser.parse_args()


def main(args):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        checkpoint = torch.load(args.ronin, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.ronin)

    network = get_model()

    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.ronin, device))

    all_sequence = glob(osp.join(args.dataset, '*'))
    for sequence in all_sequence:
        wifi, features = ronin_features(sequence)

        inputs = {}
        inputs['feature'] = []
        inputs['timestamp'] = []
        feature = features['feature']
        timestamp = features['timestamp']
        for i in range(0, len(feature)-RONIN_WINDOW_SIZE, RONIN_STEP_SIZE):
            inputs['feature'].append(feature[i:i+200].astype(np.float32).T)
            inputs['timestamp'].append(timestamp[i+RONIN_STEP_SIZE])
        inputs['feature'] = np.array(inputs['feature'])
        inputs['timestamp'] = np.array(inputs['timestamp'])

        velocity = []
        if len(inputs['feature']) >= RONIN_BATCH_SIZE:
            for i in range(np.floor(len(inputs['feature'])/RONIN_BATCH_SIZE).astype('int64')):
                in_x = torch.tensor(inputs['feature'][(i*RONIN_BATCH_SIZE):(i*RONIN_BATCH_SIZE)+RONIN_BATCH_SIZE])
                velocity.extend(list(network(in_x.to(device)).cpu().detach().numpy()))
            else:
                in_x = torch.tensor(inputs['feature'][((i+1)*RONIN_BATCH_SIZE):])
                velocity.extend(list(network(in_x.to(device)).cpu().detach().numpy()))
        else:
            in_x = torch.tensor(inputs['feature'])
            velocity.extend(list(network(in_x.to(device)).cpu().detach().numpy()))

        trajectory = []
        for v in velocity:
            trajectory.append(np.array(v) * (RONIN_INTERVAL*RONIN_STEP_SIZE))
            if len(trajectory)>1:
                trajectory[-1] += trajectory[-2]
        trajectory = np.array(trajectory)

        wifi_timestamp = [w['timestamp'] for w in wifi]
        wifi_location = interpolate_vector_linear(trajectory, inputs['timestamp'], wifi_timestamp)
        for i in range(len(wifi)):
            wifi[i]['location'] = wifi_location[i]

        # with open(osp.join(sequence, 'wifi.pickle'), 'wb') as fw:
        #     pickle.dump(wifi, fw)

        x = trajectory[:, 0]
        y = trajectory[:, 1]
        plt.plot(x, y)
        for w in wifi:
            plt.scatter(w['location'][0], w['location'][1])
        plt.show()


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    args = parse_args()
    main(args)