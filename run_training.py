import tensorflow as tf
from src.gan2 import MusicGAN
from src.music_dataset import MusicDataset

data = MusicDataset('../MusicMats/smallest_training/')
gan = MusicGAN('./new_check/','./new_out/', True)
gan.train(data, 10000)
