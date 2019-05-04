import pypianoroll
import os

def midi_to_sparse_matrix(input_path, output_path):
    proll = pypianoroll.parse(input_path)
    proll.save(output_path)


def convert_all_midis_in_dir(input_dir_path, output_dir_path):
    files = os.listdir(input_dir_path)
    for file in files:
        ofile = file.split('.')[0] + '.npz'
        ipath = os.path.join(input_dir_path, file)
        opath = os.path.join(output_dir_path, ofile)
        midi_to_sparse_matrix(ipath, opath)