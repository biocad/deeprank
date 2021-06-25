from deeprank.learn import DataSet, NeuralNet
from deeprank.learn.model3d import cnn_class
import sys
import os
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import random

DATABASE_PATH = ""

path_to_hdf5_complexes = sys.argv[1]
path_to_raw_complexes = sys.argv[2]
path_to_model = sys.argv[3]
output_dir = sys.argv[4]

TEST_COMPLEXES_FILE = 'test_complexes.txt'


def parse_complex_name(cplx_name):
    pdb_id, names = cplx_name.split('_')
    first_names, second_names = names.split('|')
    chains1, chains2 = first_names.split('+'), second_names.split('+')
    return chains1, chains2


with open(TEST_COMPLEXES_FILE, 'r') as f:
    test_names = [cplx.strip() for cplx in f.readlines()]

test_names_set = set(test_names)

database = [os.path.join(path_to_hdf5_complexes, f) for f in os.listdir(path_to_hdf5_complexes) \
            if not f.endswith(".pckl") and f[:-5] in test_names_set]



complexes_names = [dirname for dirname in os.listdir(path_to_raw_complexes) if dirname in test_names_set]

assert len(database) == len(complexes_names)

chains_dict = {cplx: (parse_complex_name(cplx)) for cplx in complexes_names}

# test_database = [os.path.join(path_to_hdf5_complexes, f) for f in test_names]


grid_info = {'number_of_points': [30, 30, 30],
             'resolution': [1., 1., 1.],
             'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
             }

print("Global test!")

data_set = DataSet(train_database=database,
                   grid_shape=(30, 30, 30),
                   grid_info=grid_info,
                   select_target='BIN_CLASS',
                   normalize_features=True,
                   normalize_targets=False,
                   clip_features=True,
                   chain_mappings=chains_dict,
                   mapfly=False,
                   use_rotation=0,
                   process=False)

model = NeuralNet(data_set=data_set,
                  model=cnn_class,
                  cuda=True,
                  task='class',
                  pretrained_model=path_to_model,
                  outdir=output_dir)

model.test()

print("Local tests for each complex")

for cplx_hdf5, cplx_name in zip(database, complexes_names):
    data_set = DataSet(train_database=[cplx_hdf5],
                       grid_shape=(30, 30, 30),
                       grid_info=grid_info,
                       select_target='BIN_CLASS',
                       normalize_features=True,
                       normalize_targets=False,
                       clip_features=True,
                       chain_mappings=chains_dict,
                       mapfly=False,
                       use_rotation=0,
                       process=False)

    model = NeuralNet(data_set=data_set,
                      model=cnn_class,
                      cuda=True,
                      task='class',
                      pretrained_model=path_to_model,
                      outdir=os.path.join(output_dir, cplx_name))
    model.test()