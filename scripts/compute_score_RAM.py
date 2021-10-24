import os
from deeprank.generate import DataGeneratorRAM
from deeprank.learn import DataSetForPretrainedModelRAM, NeuralNet
from deeprank.learn.model3d import cnn_class


import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable

import numpy as np

import sys


class NeuralNetPredictor(NeuralNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_scores(self):
        self.load_nn_params()
        self.net.train(False)
        torch.set_grad_enabled(False)
        loader = data_utils.DataLoader(self.data_set)

        data = {'log_scores': [], 'mol': []}

        for d in loader:
            inputs = d['feature']
            mol = d['mol']

            if self.cuda:
                inputs = inputs.cuda(non_blocking=True)

            # get the varialbe as float by default
            inputs = Variable(inputs).float()
            outputs = self.net(inputs)

            log_scores = F.log_softmax(outputs, dim=1)

            if self.cuda:
                data['log_scores'] += log_scores.data.cpu().numpy().tolist()
            else:
                data['log_scores'] += log_scores.data.numpy().tolist()

            molname = mol[0]
            data['mol'] += [(f, m) for f, m in zip(fname, molname)]

        data['log_scores'] = np.array(data['log_scores'])

        return data

PATH_TO_COMPLEX = '/home/gpu03/projects_jbremote/deeprank/test_apply_deeprank_model/1adq_H+L|A/'
PATH_TO_MODEL = sys.argv[1]

chains1 = ['H', 'L']
chains2 = ['A']

complex_data = DataGeneratorRAM(pdb_source=PATH_TO_COMPLEX,
                                pssm_source='.',
                                compute_features=['deeprank.features.AtomicFeature',
                                                       'deeprank.features.ResidueDensity',
                                                       'deeprank.features.BSA'],
                                compute_targets=['deeprank.targets.identity_target'],
                                chain1=chains1,
                                chain2=chains2,
                                hdf5=PATH_TO_COMPLEX)

complex_data.create_database()

grid_info = {'number_of_points': [30, 30, 30],
             'resolution': [1., 1., 1.],
             'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
            }

complex_data.map_features(grid_info, try_sparse=True, prog_bar=False)

# here comes DataSet

data_set = DataSetForPretrainedModelRAM(train_database=complex_data,
                                         grid_shape=(30, 30, 30),
                                         grid_info=grid_info,
                                         select_target='BIN_CLASS',
                                         normalize_features=True,
                                         normalize_targets=False,
                                         clip_features=True,
                                         chain_mappings=None,
                                         mapfly=False,
                                         use_rotation=0,
                                         process=False)

# and here -- NeuralNet

model = NeuralNetPredictor(data_set=data_set,
                               model=cnn_class,
                               cuda=True,
                               task='class',
                               pretrained_model=PATH_TO_MODEL)

predicted_scores = model.predict_scores()


print("Database created successfully!")