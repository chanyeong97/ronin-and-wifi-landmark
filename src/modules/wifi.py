import numpy as np
import pickle

from glob import glob
from os import path as osp

from config import *


class Wifi:
    def __init__(self, train, validation=None, test=None):
        self.train_data = {}
        self.validation_data = {}
        self.test_data = {}
        if not test:
            if validation:
                for sequence in glob(osp.join(validation, '*')):
                    self.validation_data[osp.basename(sequence)] = self.load_data(sequence)
                for sequence in glob(osp.join(train, '*')):
                    self.train_data[osp.basename(sequence)] = self.load_data(sequence)
            else:
                for sequence in glob(osp.join(train, '*')):
                    self.train_data[osp.basename(sequence)] = self.load_data(sequence)
                self.split_data()
        else:
            for sequence in glob(osp.join(test, '*')):
                self.test_data[osp.basename(sequence)] = self.load_data(sequence)

    def load_data(self, sequence):
        with open(osp.join(sequence, 'wifi.pickle'), 'rb') as fr:
            wifi = pickle.load(fr)
        return wifi

    def split_data(self):
        for sequence in self.train_data.keys():
            sequence_size = len(self.train_data[sequence])
            index = np.random.choice(sequence_size, int(sequence_size*VALIDATION_SIZE), replace=False)
            self.test_data[sequence] = []
            for i in index:
                self.test_data[sequence].append(self.train_data[sequence][i])
            for i in range(len(self.test_data[sequence])):
                self.train_data[sequence].remove(self.test_data[sequence][i])

    def load_bssid(self, path):
        with open(path, 'rb') as fr:
            self.bssid = pickle.load(fr)

    def select_bssid(self):
        self.bssid = []
        self.bssid_count = []
        for sequence in self.train_data.values():
            for wifi in sequence:
                for bssid in wifi['bssid']:
                    if bssid in self.bssid:
                        self.bssid_count[self.bssid.index(bssid)] += 1
                    else:
                        self.bssid.append(bssid)
                        self.bssid_count.append(1)
        
        for i in range(len(self.bssid)-1, -1, -1):
            if self.bssid_count[i] < BSSID_THRESHOLD:
                del self.bssid[i]
                del self.bssid_count[i]
    
    def get_bssid(self):
        return self.bssid

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_validation_data(self):
        return self.validation_data

