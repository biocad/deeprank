from deeprank.learn import DataSet, NeuralNet
from deeprank.learn.modelGenerator import *
from deeprank.learn.model3d import cnn_class
import sys
import os
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import random

print("Hello!")

DATABASE_PATH = ""

path_to_hdf5_complexes = sys.argv[1]
path_to_raw_complexes = sys.argv[2]
name_of_output_file = sys.argv[3]
NUM_WORKERS = int(sys.argv[4])
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
            if not f.endswith(".pckl") and not f[:-5] in test_names_set]

complexes_names = [dirname for dirname in os.listdir(path_to_raw_complexes)]

near_native = 0
total = 0

for cplx in tqdm(database, total=len(database)):
    if cplx.endswith('.hdf5'):
        h5 = h5py.File(cplx, 'r')
        for mol in h5.keys():
            near_native += h5[mol]['targets']['BIN_CLASS'][()]
            total += 1

print(f"near-native poses: {near_native / total}")
weight0 = near_native / total
weight1 = 1 - weight0

chains_dict = {cplx: (parse_complex_name(cplx)) for cplx in complexes_names}

grid_info = {'number_of_points': [30, 30, 30],
             'resolution': [1., 1., 1.],
             'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
             }

train_database, testvalid_database = train_test_split(database, test_size=0.1)

data_set = DataSet(train_database=train_database,
                   valid_database=testvalid_database,
                   grid_shape=(30, 30, 30),
                   grid_info=grid_info,
                   select_target='BIN_CLASS',
                   normalize_features=True,
                   normalize_targets=False,
                   pair_chain_feature=np.add,
                   chain_mappings=chains_dict,
                   mapfly=False)

model = NeuralNet(data_set=data_set,
                  model=cnn_class,
                  model_type='3d',
                  task='class',
                  class_weights=torch.FloatTensor([weight0, weight1]).cuda(),
                  cuda=True)

model.train(nepoch=50,
            train_batch_size=32,
            num_workers=NUM_WORKERS,
            hdf5=name_of_output_file)
