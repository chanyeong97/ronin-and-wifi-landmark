import numpy as np

from os import path as osp


NS2S = 1e9
MS2S = 1e6
TEST_SIZE = 0.3


### bssid
BSSID_THRESHOLD = 20


### ronin
RONIN_STEP_SIZE             = 10
RONIN_WINDOW_SIZE           = 200
RONIN_INTERVAL              = 1. / 200
RONIN_BATCH_SIZE            = 1024


### autoencoder
AUTOENCODER_DROPOUT         = 0.3
AUTOENCODER_BATCH_SIZE      = 1024
AUTOENCODER_LEARNING_RATE   = 0.0001
AUTOENCODER_EPOCHS          = 25000
AUTOENCODER_MAX_TO_KEEP     = 5

ENCODING_LAYER_1            = 300
ENCODING_LAYER_2            = 300
ENCODING_OUTPUT_LAYER       = 30
DECODING_LAYER_1            = 300
DECODING_LAYER_2            = 300


### landmark
LANDMARK_DROPOUT            = 0.3
LANDMARK_BATCH_SIZE         = 1024
LANDMARK_LEARNING_RATE      = 0.00003
LANDMARK_EPOCHS             = 15000
LANDMARK_MAX_TO_KEEP        = 5

LANDMARK_LAYER_1            = 300
LANDMARK_LAYER_2            = 300
LANDMARK_LAYER_3            = 300
LANDMARK_LAYER_4            = 300
LANDMARK_OUTPUT_LAYER       = 4

LANDMARK_POSITION = {
    '1':        np.array([[0., 0.]]), 
    '2to1':     np.array([[8., 0.]]), 
    '2':        np.array([[31., 0.]]), 
    '1to2':     np.array([[25., 0.]]), 
    '3to2':     np.array([[31., 5.]]), 
    '3':        np.array([[33., 28.]]), 
    '2to3':     np.array([[33., 24.]]), 
}
