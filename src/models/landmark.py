from tensorflow.keras.models import Model

from config import *


class Landmark(Model):
    def __init__(self):
        super(Landmark, self).__init__()
