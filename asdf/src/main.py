import config

from src.modules.dataset import load_wifi_dataset





if __name__ == '__main__':
    wifi = load_wifi_dataset()

    ### autoencoder


    ### landmark detection





    ### wifi network
    """
    각 와이파이간 거리를 이용해서 와이파이 지문을 얻는 데이터셋1, DNN모델1

    위에서 학습시킨 모델을 사용해서 와이파이 위치를 추측하고 ronin을 이용해서 얻은 이동 거리를 토대로
    와이파이를 통해 얻은 위치를 보정할 데이터셋1, RNN모델1
    """