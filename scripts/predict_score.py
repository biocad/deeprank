import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)

from deeprank.learn import DataSet, NeuralNet
from deeprank.learn.model3d import cnn_class
from deeprank.generate import *
from deeprank.config import logger
from deeprank.tools import sparse

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable
import h5py

import numpy as np
import sys
import os
import shutil
import warnings

class DataSetForPretrainedModel(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dataset(self):
        """Process the data set.

        Done by default. However must be turned off when one want to
        test a pretrained model. This can be done by setting
        ``process=False`` in the creation of the ``DataSet`` instance.
        """

        logger.info('\n')
        logger.info('=' * 40)
        logger.info('=\t DeepRank Data Set')
        logger.info('=')
        logger.info('=\t Training data')
        for f in self.train_database:
            logger.info(f'=\t -> {f}')
        logger.info('=')
        if self.valid_database:
            logger.info('=\t Validation data')
            for f in self.valid_database:
                logger.info(f'=\t -> {f}')
        logger.info('=')
        if self.test_database:
            logger.info('=\t Test data')
            for f in self.test_database:
                logger.info(f'=\t -> {f}')
        logger.info('=')
        logger.info('=' * 40 + '\n')
        sys.stdout.flush()

        # check if the files are ok
        self.check_hdf5_files(self.train_database)

        if self.valid_database:
            self.valid_database = self.check_hdf5_files(
                self.valid_database)

        if self.test_database:
            self.test_database = self.check_hdf5_files(
                self.test_database)

        # create the indexing system
        # alows to associate each mol to an index
        # and get fname and mol name from the index
        self.create_index_molecules()

        # get the actual feature name
        if self.mapfly:
            self.get_raw_feature_name()
        else:
            self.get_mapped_feature_name()

        # get the pairing
        self.get_pairing_feature()

        # get grid shape
        self.get_grid_shape()

        # get the input shape
        self.get_input_shape()

        logger.info('\n')
        logger.info("   Data Set Info:")
        logger.info(
            f'   Augmentation       : {self.use_rotation} rotations')
        logger.info(
            f'   Training set       : {self.ntrain} conformations')
        logger.info(
            f'   Validation set     : {self.nvalid} conformations')
        logger.info(
            f'   Test set           : {self.ntest} conformations')
        logger.info(f'   Number of channels : {self.input_shape[0]}')
        logger.info(f'   Grid Size          : {self.data_shape[1]}, '
                    f'{self.data_shape[2]}, {self.data_shape[3]}')
        sys.stdout.flush()


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

            fname, molname = mol[0], mol[1]
            data['mol'] += [(f, m) for f, m in zip(fname, molname)]

        data['log_scores'] = np.array(data['log_scores'])

        return data


def parse_complex_name(cplx_name):
    pdb_id, names = cplx_name.split('_')
    first_names, second_names = names.split('|')
    ch1, ch2 = first_names.split('+'), second_names.split('+')
    return ch1, ch2



def main():
    warnings.filterwarnings("ignore")

    PATH_TO_COMPLEX = sys.argv[1]
    PATH_TO_MODEL = sys.argv[2]
    # POSES_COUNT = sys.argv[3]
    COMPLEXES_PREFIX = "predicted_complex_"

    indices_dict = {COMPLEXES_PREFIX + str(i + 1): i for i in range(len(os.listdir(PATH_TO_COMPLEX)))}

    complex_name = os.path.basename(os.path.normpath(PATH_TO_COMPLEX))
    hdf5_out_dir = os.path.abspath(os.path.join(PATH_TO_COMPLEX, os.pardir))
    hdf5_file = os.path.join(hdf5_out_dir, f"{complex_name}.hdf5")

    chains1, chains2 = parse_complex_name(complex_name)

    complex_data = DataGenerator(pdb_source=os.path.join(PATH_TO_COMPLEX),
                                 pssm_source='.',
                                 compute_features=['deeprank.features.AtomicFeature',
                                                   'deeprank.features.ResidueDensity',
                                                   'deeprank.features.BSA'],
                                 compute_targets=['deeprank.targets.identity_target'],
                                 chain1=chains1,
                                 chain2=chains2,
                                 hdf5=hdf5_file)
    complex_data.create_database(prog_bar=False)

    grid_info = {'number_of_points': [30, 30, 30],
                 'resolution': [1., 1., 1.],
                 'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
                 }

    complex_data.map_features(grid_info, try_sparse=True, prog_bar=False)

    data_set = DataSetForPretrainedModel(train_database=hdf5_file,
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

    model = NeuralNetPredictor(data_set=data_set,
                               model=cnn_class,
                               cuda=True,
                               task='class',
                               pretrained_model=PATH_TO_MODEL)

    predicted_scores = model.predict_scores()

    score_index_pairs = [(predicted_scores['log_scores'][i][1], indices_dict[predicted_scores['mol'][i][1]]) for i in
                         range(len(predicted_scores['mol']))]
    sorted_pairs = sorted(score_index_pairs, key=lambda pair: -pair[0])

    result = "\n".join(map(str, [pair[1] for pair in sorted_pairs]))

    #cleanup
    shutil.rmtree(PATH_TO_COMPLEX)
    os.remove(hdf5_file)

    print(result)

main()
