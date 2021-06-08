import sys
from multiprocessing import Pool
import h5py
import os
from tqdm import tqdm

path_to_hdf5_complexes = sys.argv[1]
NUM_WORKERS = int(sys.argv[2])
POOL_SIZE = NUM_WORKERS
TEST_COMPLEXES_FILE = 'test_complexes.txt'

with open(TEST_COMPLEXES_FILE, 'r') as f:
    test_names = [cplx.strip() for cplx in f.readlines()]

test_names_set = set(test_names)

database = [os.path.join(path_to_hdf5_complexes, f) for f in os.listdir(path_to_hdf5_complexes)
            if f.endswith(".hdf5") and not f[:-5] in test_names_set]


def calculate_for_complex(cplx):
    near_native = 0
    total = 0
    h5 = h5py.File(cplx, 'r')
    for mol in h5.keys():
        near_native += h5[mol]['targets']['BIN_CLASS'][()]
        total += 1
    return near_native, total

results = []
with Pool(POOL_SIZE) as p:
    max_ = len(database)
    with tqdm(total=max_) as pbar:
        for r in p.imap_unordered(calculate_for_complex, database):
            results.append(r)
            pbar.update()

total_near_native = 0
total_total = 0

for nn, tot in results:
    total_near_native += nn
    total_total += tot

print(f"Fraction of near native poses {total_near_native / total_total}")
