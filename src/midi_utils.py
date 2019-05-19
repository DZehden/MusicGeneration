import pypianoroll
import os
import numpy as np


def ndarray_to_midi(arr, output_dir_path, name='output'):
    """
    Write a single tracks array representation to midi file

    :param arr: ndarray
        pianoroll ndarray to convrert
    :param output_dir_path: str
        directory to write midi
    :param name: str
        name of file to write
    :return: None
    """
    trk = pypianoroll.Track(pianoroll=arr, program=0, name=name)
    multi = pypianoroll.Multitrack(tracks=[trk])
    pypianoroll.write(multi, output_dir_path)

def ndarray_to_npz(arr, output_dir_path, name='output'):
    """
    Write a single tracks array representation to npz file

    :param arr: ndarray
        pianoroll ndarray to convrert
    :param output_dir_path: str
        directory to write midi
    :param name: str
        name of file to write
    :return: None
    """
    trk = pypianoroll.Track(pianoroll=arr, program=0, name=name)
    multi = pypianoroll.Multitrack(tracks=[trk])
    pypianoroll.save(output_dir_path, multi)


def load_pianoroll(input_path):
    """
    Convenience method to get pianoroll

    :param input_path: str
         path to midi
    :return: pypianoroll.MultiTrack
    """
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


def convert_midis_to_npz(root_dir, output_dir, samples_per_track=1, beats_per_sample=16):
    """
    Recursively creates pianoroll samples starting at a root directory and outputs them to a single output directory

    :param root_dir: str
        path to root of file tree to search for midi files
    :param output_dir: str
        path to output directory
    :param samples_per_track: int
        number of samples to extract from each midi found
    :param beats_per_sample: int
        number of beats to include in each sample
    :return: None
    """
    def get_sample_name(songfile, sid):
        return songfile.split('.')[0] + '_sample_{0}'.format(sid)

    def get_sample_filename(songfile, sid):
        return songfile.split('.')[0] + '_sample_{0}.npz'.format(sid)

    time_slices_to_sample = 24 * beats_per_sample
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            fullpath = os.path.join(root, file)
            f = file.lower()
            if '.mid' in f:
                try:
                    roll = pypianoroll.parse(fullpath)
                except Exception as e:
                    print(e)
                    print('Unable to parse {0}: continuing...'.format(fullpath))
                    continue
                if len(roll.tracks) == 1:
                    mat = roll.tracks[0].pianoroll
                    max_start_idx = ((mat.shape[0] - time_slices_to_sample) // 24) * 24
                    if max_start_idx < 0:
                        continue
                    for sample_num in range(samples_per_track):
                        sample_start_idx = np.random.randint(0, max_start_idx)
                        sample = mat[sample_start_idx: sample_start_idx + time_slices_to_sample]
                        sample[sample > 0] = 127
                        sample_track = pypianoroll.Track(pianoroll=sample, name=get_sample_name(file, sample_num))
                        sample_multi = pypianoroll.Multitrack(tracks=[sample_track])
                        pypianoroll.save(os.path.join(output_dir, get_sample_filename(file, sample_num)), sample_multi)


def npz_to_midi(path_to_npz, output_dir):
    """
    Converts an npz file to a midi

    :param path_to_npz: str
         path to npz file to convert
    :param output_dir:
         path to directory where midi will be stored
    :return: None
    """
    filename = path_to_npz.split('/')[-1].split('.')[0]
    roll = pypianoroll.load(path_to_npz)
    pypianoroll.write(roll, os.path.join(output_dir, filename + '.mid'))
