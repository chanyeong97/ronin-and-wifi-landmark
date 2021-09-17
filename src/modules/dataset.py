import os
import numpy as np
import quaternion
import pandas as pd
import scipy
import config


def interpolate_vector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def interpolate_quaternion_linear(data, ts_in, ts_out):
    assert np.amin(ts_in) <= np.amin(ts_out), 'Input time range must cover output time range'
    assert np.amax(ts_in) >= np.amax(ts_out), 'Input time range must cover output time range'
    pt = np.searchsorted(ts_in, ts_out)
    d_left = quaternion.from_float_array(data[pt - 1])
    d_right = quaternion.from_float_array(data[pt])
    ts_left, ts_right = ts_in[pt - 1], ts_in[pt]
    d_out = quaternion.quaternion_time_series.slerp(d_left, d_right, ts_left, ts_right, ts_out)
    return quaternion.as_float_array(d_out)


def process_data_source(raw_data, output_time, method):
    input_time = raw_data[:, 0]
    if method == 'vector':
        output_data = interpolate_vector_linear(raw_data[:, 1:], input_time, output_time)
    elif method == 'quaternion':
        assert raw_data.shape[1] == 5
        output_data = interpolate_quaternion_linear(raw_data[:, 1:], input_time, output_time)
    else:
        raise ValueError('Interpolation method must be "vector" or "quaternion"')
    return output_data


def compute_output_time(all_sources, sample_rate=200):
    interval = 1. / sample_rate
    min_t = max([data[0, 0] for data in all_sources.values()]) + interval
    max_t = min([data[-1, 0] for data in all_sources.values()]) - interval
    return np.arange(min_t, max_t, interval)


def load_ronin_features(dataset):
    wifi = {}
    features = {}
    for sequence in dataset:
        wifi_pd = pd.read_csv( os.path.join(sequence, 'wifi.txt'), delimiter='\t', names=['timestamp', 'bssid', 'rssi'])
        wifi_list = wifi_pd.values.tolist()
        sequence_name = os.path.basename(sequence)
        wifi[sequence_name] = [{'timestamp': wifi_list[0][0] / config.MS2S}]
        bssid = []
        level = []
        for data in wifi_list:
            timestamp = data[0] / config.MS2S
            if not wifi[sequence_name][-1]['timestamp'] == timestamp:
                wifi[sequence_name][-1]['bssid'] = bssid
                wifi[sequence_name][-1]['level'] = level
                wifi[sequence_name].append({'timestamp': timestamp})
                bssid = []
                level = []
            bssid.append(data[1])
            level.append(data[2])
        else:
            wifi[sequence_name][-1]['bssid'] = bssid
            wifi[sequence_name][-1]['level'] = level
        
        gyro = np.loadtxt(os.path.join(sequence, 'gyro.txt'))
        acce = np.genfromtxt(os.path.join(sequence, 'acce.txt'))
        gyro_uncalib = np.genfromtxt(os.path.join(sequence, 'gyro_uncalib.txt'))[:, :4]
        game_rv = np.genfromtxt(os.path.join(sequence, 'game_rv.txt'))

        acce[:, 0] /= config.NS2S
        gyro[:, 0] /= config.NS2S
        gyro_uncalib[:, 0] /= config.NS2S
        game_rv[:, 0] /= config.NS2S

        all_sources = {
            'acce': acce,
            'gyro': gyro,
            'gyro_uncalib': gyro_uncalib,
            'game_rv': game_rv
        }
        output_time = compute_output_time(all_sources)

        acce = process_data_source(acce, output_time, 'vector')
        gyro = process_data_source(gyro, output_time, 'vector')
        gyro_uncalib = process_data_source(gyro_uncalib, output_time, 'vector')
        game_rv = process_data_source(game_rv[:, [0, 4, 1, 2, 3]], output_time, 'quaternion')
        init_gyro_bias = gyro_uncalib[0] - gyro[0]
        gyro_uncalib -= init_gyro_bias

        ori_q = quaternion.from_float_array(game_rv)
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro_uncalib.shape[0], 1]), gyro_uncalib], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
        features[sequence_name] = {}
        features[sequence_name]['feature'] = np.concatenate([glob_gyro, glob_acce], axis=1)
        features[sequence_name]['timestamp'] = output_time

    return wifi, features