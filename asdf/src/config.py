import os

from glob import glob


NS2S = 1e9
MS2S = 1e6

TEST_DATASET_PATH = os.path.join('dataset', 'test')
LANDMARK_TRAIN_DATASET_PATH = os.path.join('dataset', 'landmark')

RONIN_RESNET_MODEL_PATH = os.path.join('trained_models', 'ronin_resnet', 'checkpoint_gsn_latest.pt')
AUTOENCODER_MODEL_PATH = os.path.join('trained_models', 'autoencoder', 'checkpoint')

def get_train_dataset():
    return glob(os.path.join(TRAIN_DATASET_PATH, '*'))