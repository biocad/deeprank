from deeprank.learn import DataSet, NeuralNet
from deeprank.learn.modelGenerator import *
from deeprank.learn.model3d import cnn_class
import sys
import os
import numpy as np

DATABASE_PATH = ""

path_to_hdf5_complexes = sys.argv[1]
path_to_raw_complexes = sys.argv[2]


def parse_complex_name(cplx_name):
    pdb_id, names = cplx_name.split('_')
    first_names, second_names = names.split('|')
    chains1, chains2 = first_names.split('+'), second_names.split('+')
    return chains1, chains2


database = [os.path.join(path_to_hdf5_complexes, f) for f in os.listdir(path_to_hdf5_complexes)]
complexes_names = [dirname for dirname in os.listdir(path_to_raw_complexes)]

chains_dict = {cplx : (parse_complex_name(cplx)) for cplx in complexes_names}

grid_info = {'number_of_points': [30, 30, 30],
                     'resolution': [1., 1., 1.],
                     'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
                     }

data_set = DataSet(train_database=database,
                   valid_database=None,
                   test_database=None,
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
                  task='class')

model.train(nepoch=5,
            train_batch_size=5,
            num_workers=1,
            hdf5='train_example')
