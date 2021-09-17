import os

from glob import glob


NS2S = 1e9
MS2S = 1e6

TRAIN_DATASET_PATH = os.path.join('dataset', 'train')

RONIN_RESNET_MODEL_PATH = os.path.join('models', 'ronin_resnet', 'checkpoint_gsn_latest.pt')
AUTOENCODER_MODEL_PATH = os.path.join('models', 'autoencoder', 'checkpoint')

def get_train_dataset():
    return glob(os.path.join(TRAIN_DATASET_PATH, '*'))