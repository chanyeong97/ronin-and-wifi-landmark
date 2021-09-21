import scipy
import numpy as np
import quaternion
import pandas as pd

from os import path as osp

from config import *


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


def ronin_features(sequence):
    wifi = []
    features = {}

    gyro = np.genfromtxt(osp.join(sequence, 'gyro.txt'))
    acce = np.genfromtxt(osp.join(sequence, 'acce.txt'))
    gyro_uncalib = np.genfromtxt(osp.join(sequence, 'gyro_uncalib.txt'))[:, :4]
    game_rv = np.genfromtxt(osp.join(sequence, 'game_rv.txt'))

    acce[:, 0] /= NS2S
    gyro[:, 0] /= NS2S
    gyro_uncalib[:, 0] /= NS2S
    game_rv[:, 0] /= NS2S

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

    features['feature'] = np.concatenate([glob_gyro, glob_acce], axis=1)
    features['timestamp'] = output_time

    wifi_pd = pd.read_csv(osp.join(sequence, 'wifi.txt'), delimiter='\t', names=['timestamp', 'bssid', 'level'])
    wifi_list = wifi_pd.values.tolist()
    for v in wifi_list:
        timestamp = v[0] / MS2S
        bssid = v[1]
        level = v[2]
        if output_time[0] < timestamp and output_time[-(RONIN_WINDOW_SIZE+RONIN_STEP_SIZE)] > timestamp:
            if len(wifi) == 0 or not wifi[-1]['timestamp'] == timestamp:
                wifi.append({
                    'timestamp': timestamp,
                    'bssid': [bssid],
                    'level': [level]})
            else:
                wifi[-1]['bssid'].append(bssid)
                wifi[-1]['level'].append(level)
    return wifi, features
