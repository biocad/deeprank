import sys
import os
from tqdm import *
import shutil
# from mpi4py import MPI
from multiprocessing import Pool

# comm = MPI.COMM_WORLD

from deeprank.generate import *

DECOY_SUBDIR = 'decoys'
NATIVE_SUBDIR = 'native'

input_data_path = sys.argv[1]
hdf5_out_dir = sys.argv[2]
POOL_SIZE = int(sys.argv[3])

if os.path.exists(hdf5_out_dir):
    shutil.rmtree(hdf5_out_dir)

if not os.path.exists(hdf5_out_dir):
    os.makedirs(hdf5_out_dir)

def parse_complex_name(cplx_name):
    pdb_id, names = cplx_name.split('_')
    first_names, second_names = names.split('|')
    ch1, ch2 = first_names.split('+'), second_names.split('+')
    return ch1, ch2


# for i, cplx in enumerate(tqdm(os.listdir(input_data_path), total=len(os.listdir(input_data_path)))):
#     try:
#         chains1, chains2 = parse_complex_name(cplx)
#         # TBD generate features
#         complex_data = DataGenerator(pdb_source=os.path.join(input_data_path, cplx, DECOY_SUBDIR),
#                                      pdb_native=os.path.join(input_data_path, cplx, NATIVE_SUBDIR),
#                                      pssm_source='.',
#                                      compute_features=['deeprank.features.AtomicFeature',
#                                                        'deeprank.features.ResidueDensity',
#                                                        'deeprank.features.BSA'],
#                                      compute_targets=['deeprank.targets.binary_class'],
#                                      chain1=chains1,
#                                      chain2=chains2,
#                                      hdf5=os.path.join(hdf5_out_dir, f"{cplx}.hdf5"),
#                                      mpi_comm=comm)
#         complex_data.create_database(prog_bar=True)
#         grid_info = {'number_of_points': [30, 30, 30],
#                      'resolution': [1., 1., 1.],
#                      'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
#                     }
#         complex_data.map_features(grid_info, try_sparse=True, prog_bar=True)
#         print(f"generated for {cplx} [{i+1}/{len(os.listdir(input_data_path))}]")
#     except Exception:
#         continue

def calculate(cplx):
    try:
        chains1, chains2 = parse_complex_name(cplx)
        # TBD generate features
        complex_data = DataGenerator(pdb_source=os.path.join(input_data_path, cplx, DECOY_SUBDIR),
                                     pdb_native=os.path.join(input_data_path, cplx, NATIVE_SUBDIR),
                                     pssm_source='.',
                                     compute_features=['deeprank.features.AtomicFeature',
                                                       'deeprank.features.ResidueDensity',
                                                       'deeprank.features.BSA'],
                                     compute_targets=['deeprank.targets.binary_class'],
                                     chain1=chains1,
                                     chain2=chains2,
                                     hdf5=os.path.join(hdf5_out_dir, f"{cplx}.hdf5"))
                                     # data_augmentation=5)
        complex_data.create_database(prog_bar=False)
        grid_info = {'number_of_points': [30, 30, 30],
                     'resolution': [1., 1., 1.],
                     'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
                     }
        complex_data.map_features(grid_info, try_sparse=True, prog_bar=False)
    except Exception:
        pass


complexes_list = os.listdir(input_data_path)

results = []
with Pool(POOL_SIZE) as p:
    max_ = len(complexes_list)
    with tqdm(total=max_) as pbar:
        for r in p.imap_unordered(calculate, complexes_list):
            results.append(r)
            pbar.update()