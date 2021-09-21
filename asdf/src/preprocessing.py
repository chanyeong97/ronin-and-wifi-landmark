import torch
import config
import matplotlib.pyplot as plt
import os
import pickle

from src.models.ronin import *
from src.modules.dataset import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
_input_channel, _output_channel = 6, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}


def get_model():
    network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                        base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    return network


if __name__ == '__main__':
    dataset = config.get_train_dataset()
    wifi, features = load_ronin_features(dataset)

    inputs = {}
    step_size = 10
    window_size = 200
    batch_size = 1024
    interval = 1. / 200
    for sequence_name in features.keys():
        inputs[sequence_name] = {}
        inputs[sequence_name]['feature'] = []
        inputs[sequence_name]['timestamp'] = []
        feature = features[sequence_name]['feature']
        timestamp = features[sequence_name]['timestamp']
        for i in range(0, len(feature)-window_size, step_size):
            inputs[sequence_name]['feature'].append(feature[i:i+200].astype(np.float32).T)
            inputs[sequence_name]['timestamp'].append(timestamp[i+step_size])
        inputs[sequence_name]['feature'] = np.array(inputs[sequence_name]['feature'])
        inputs[sequence_name]['timestamp'] = np.array(inputs[sequence_name]['timestamp'])

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        checkpoint = torch.load(config.RONIN_RESNET_MODEL_PATH, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(config.RONIN_RESNET_MODEL_PATH)

    network = get_model()
    
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)

    for sequence_name in inputs.keys():
        pred = []
        if len(inputs[sequence_name]['feature']) >= batch_size:
            for i in range(np.floor(len(inputs[sequence_name]['feature'])/batch_size).astype('int64')):
                in_x = torch.tensor(inputs[sequence_name]['feature'][(i*batch_size):(i*batch_size)+batch_size])
                pred.extend(list(network(in_x.to(device)).cpu().detach().numpy()))
            else:
                in_x = torch.tensor(inputs[sequence_name]['feature'][((i+1)*batch_size):])
                pred.extend(list(network(in_x.to(device)).cpu().detach().numpy()))
        else:
            in_x = torch.tensor(inputs[sequence_name]['feature'])
            pred.extend(list(network(in_x.to(device)).cpu().detach().numpy()))

        trajectory = []
        for i in pred:
            trajectory.append(np.array(i) * (interval*step_size))
            if len(trajectory)>1:
                trajectory[-1] += trajectory[-2]

        trajectory = np.array(trajectory)
        output_timestamp = []
        for i in range(len(wifi[sequence_name])):
            if wifi[sequence_name][i]['timestamp'] > max(inputs[sequence_name]['timestamp']):
                break
            output_timestamp.append(wifi[sequence_name][i]['timestamp'])
        
        wifi_location = interpolate_vector_linear(trajectory, inputs[sequence_name]['timestamp'], output_timestamp)
        for out_t in output_timestamp:
            for wifi_t in wifi[sequence_name]:
                if wifi_t['timestamp'] == out_t:
                    wifi[sequence_name][wifi[sequence_name].index(wifi_t)]['location'] = wifi_location[output_timestamp.index(out_t)]
                    break

        for w in wifi[sequence_name]:
            if not 'location' in w:
                wifi[sequence_name].remove(w)

        with open(os.path.join(config.TRAIN_DATASET_PATH, sequence_name, 'wifi.pickle'), 'wb') as fw:
            pickle.dump(wifi[sequence_name], fw)

        # x = trajectory[:, 0]
        # y = trajectory[:, 1]
        # plt.plot(x, y)
        # plt.scatter(w[:, 0], w[:, 1])
        # plt.show()

    