import tensorflow as tf
from src.gan2 import MusicGAN
from src.music_dataset import MusicDataset
from torch.utils.data import RandomSampler

data = MusicDataset('../MusicMats/smallest_training/')
sampler = RandomSampler(data, replacement=True, num_samples=100)
gan = MusicGAN('./new_check/','./new_out/', True)
gan.train(sampler, 10000)
