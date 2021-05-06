from deeprank.learn import DataSet, NeuralNet
from deeprank.learn.modelGenerator import *
from deeprank.learn.model3d import cnn_class
import sys
import os
import numpy as np

DATABASE_PATH = ""

path_to_hdf5_complexes = sys.argv[1]

database = [os.path.join(path_to_hdf5_complexes, f) for f in os.listdir(path_to_hdf5_complexes)]

data_set = DataSet(train_database=database,
                   valid_database=None,
                   test_database=None,
                   grid_shape=(30, 30, 30),
                   select_target='BIN_CLASS',
                   normalize_features=True,
                   normalize_targets=False,
                   pair_chain_feature=np.add,
                   mapfly=False)

model = NeuralNet(data_set=data_set,
                  model=cnn_class,
                  model_type='3d',
                  task='class')

model.train(nepoch=5,
            train_batch_size=5,
            num_workers=1,
            hdf5='train_example')

