import numpy as np
from tqdm import tqdm

import cProfile, pstats
from pstats import SortKey
import io

from deeprank.generate import DataGeneratorRAM
from deeprank.learn import DataSetForPretrainedModelRAM, NeuralNet
from deeprank.learn.model3d import cnn_class

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable


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

class MyPDBComplex:

    def __init__(self, filename):
        pass

def get_pdb_data_and_coordinates(filename):
    data = []
    coords = []
    with open(filename, 'r') as fi:
        for line in fi:
            if line.startswith('ATOM'):
                data.append(line)
                #here we extract coordinates, as it is done in biopython
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    # data = np.array(data).astype('|S78')
    coords = np.array(coords)

    return data, coords

def get_rotations_biases(filepath):
    rotation_matrices = []
    biases = []
    decoy_names = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('conformation'):
                name = line.split('[')[0]
                decoy_names.append(name)
            if line.startswith('coords_bias'):
                floats = line.split(':')[1]
                bias = [float(x) for x in floats.split()]
                biases.append(bias)
            if line.startswith('rotation_rotation_matrix'):
                floats = line.split(':')[1]
                rotation = np.array([float(x) for x in floats.split()]).reshape((3,3))
                rotation_matrices.append(rotation)
    return decoy_names, np.array(biases), np.array(rotation_matrices)


def rotate_and_shift_coords(input_coords, rotation_matrices, biases):
    """

    Args:
        input_coords: np.ndarray of shape (N_atoms, 3)
        rotation_matrices: np.ndarray of shape (N_conformations, 3, 3)
        biases: np.ndarray of shape (N_conformations, 3)
    Returns:
        np.array of shape (N_conformations, N_atoms, 3)
    """
    res = np.dot(input_coords, rotation_matrices.transpose((0,2,1))) + biases
    return res

def float_to_str_pdb(x):
    return "%8.3f" % x


def set_coords(old_data, new_coords):
    if len(old_data) != new_coords.shape[0]:
        raise ValueError("Invalid dimensions! old_data and new_coords must have same length")

    new_data = []
    coords_list = list(new_coords)

    for i in range(len(old_data)):
        line = old_data[i]
        x, y, z = coords_list[i]
        new_line = line[:30] + float_to_str_pdb(x) + float_to_str_pdb(y) + float_to_str_pdb(z) + line[54:]
        new_data.append(new_line)

    return new_data

def to_pdb(data, file):
    with open(file, 'w') as f:
        f.writelines(data)

ligand_data, ligand_coords = get_pdb_data_and_coordinates('./1adq_H+L|A/prepared_schrod/1adq_ag_u.pdb')
receptor_data, _ = get_pdb_data_and_coordinates('./1adq_ab_u_fv.pdb')



names, biases, rotations = get_rotations_biases('rotations_biases.txt')


res = rotate_and_shift_coords(ligand_coords, rotations, biases).transpose((1,0,2))

print(len(ligand_data))
print(ligand_coords.shape)
print(biases.shape)
print(rotations.shape)

print(res.shape)

first_conf = res[0]

#сеттим координаты, и вот это надо передать в DeepRank. Дальше там все понятно, берем и делаем
# new_conformation = set_coords(ligand_data, first_conf)
# to_pdb(new_conformation, "test_conf.pdb")

list_of_conformations = []
for i in tqdm(range(5)):
    data = np.array(set_coords(ligand_data, res[i]) + receptor_data).astype('|S78')
    list_of_conformations.append((f"predicted_complex_{i+1}", data))

chains1 = ['H', 'L']
chains2 = ['A']

pr = cProfile.Profile()

pr.enable()
complex_data = DataGeneratorRAM(pdb_source=list_of_conformations,
                                pssm_source='.',
                                compute_features=['deeprank.features.AtomicFeature',
                                                       'deeprank.features.ResidueDensity',
                                                       'deeprank.features.BSA'],
                                compute_targets=['deeprank.targets.identity_target'],
                                chain1=chains1,
                                chain2=chains2,
                                hdf5='.')

complex_data.create_database()
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
ps.print_stats()
with open('profile_clust1.txt', 'w+') as f:
    f.write(s.getvalue())

grid_info = {'number_of_points': [30, 30, 30],
             'resolution': [1., 1., 1.],
             'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
            }

complex_data.map_features(grid_info, try_sparse=True, prog_bar=False)

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

PATH_TO_MODEL = '/home/gpu03/projects_jbremote/deeprank/sample_train/best_valid_model.pth.tar'

model = NeuralNetPredictor(data_set=data_set,
                               model=cnn_class,
                               cuda=False,
                               task='class',
                               pretrained_model=PATH_TO_MODEL)

predicted_scores = model.predict_scores()

print("success")




