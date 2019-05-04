import pypianoroll
import os

def midi_to_sparse_matrix(input_path, output_path):
    """
    Parses a midi into an npz csc file

    :param input_path: str
        absolute path of input file
    :param output_path: str
        absolute path of output directory
    :return: None
    """
    if '/' in input_path:
        fname = input_path.split('/')[-1]
    else:
        fname = input_path
    ofile = '{0}.npz'.format(fname.split('.')[0])
    opath = os.path.join(output_path, ofile)
    proll = pypianoroll.parse(input_path)
    proll.save(opath)


def parse_all_midis_in_dir(input_dir_path, output_dir_path):
    """
    Parses all midis in a directory into an npz csc file

    :param input_dir_path: str
        absolute path to directory with midi files
    :param output_dir_path: str
        absolute path to output directory
    :return: None
    """
    files = os.listdir(input_dir_path)
    for file in files:
        if '.mid' not in file:
            continue
        ofile = '{0}.npz'.format(file.split('.')[0])
        ipath = os.path.join(input_dir_path, file)
        opath = os.path.join(output_dir_path, ofile)
        midi_to_sparse_matrix(ipath, opath)

def load_pianoroll(input_path):
    return pypianoroll.parse(input_path)


def get_pianoroll_matrix(pianoroll, track=None):
    """
    Extracts a pianorolls matrix representation

    :param pianoroll: Track or Multitrack
         pianoroll object with matrix representation
    :param track: int
         track number of multitrack to extract
    :return: ndarray
    """
    ret = None
    if isinstance(pianoroll, pypianoroll.Track):
        ret = pianoroll.pianoroll
    elif isinstance(pianoroll, pypianoroll.Multitrack):
        if isinstance(track, int) and track >= 0:
            ret = pianoroll.tracks[track].pianoroll
        else:
            print('Pianoroll is multitrack: must specify a track index to extract')
    else:
        print('Pianoroll parameter is not a valid pianoroll object')
    return ret
