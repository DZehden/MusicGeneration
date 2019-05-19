from torch.utils.data import Dataset
import os
import pypianoroll

class MusicDataset(Dataset):
    """
    Dataset class for music data. Initialized with a list of filenames corresponding to
    npz files containing a song segment
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = os.listdir(data_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        if '.npz' not in self.filenames[item]:
            print('Found invalid data: {0}'.format(self.filenames[item]))
        roll = pypianoroll.load(os.path.join(self.data_dir, self.filenames[item]))
        mat = roll.tracks[0].pianoroll
        mat = mat.astype('float32')
        mat = (mat - 64.5) / 64.5
        return mat