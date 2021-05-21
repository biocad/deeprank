import sys
import os
from shutil import copyfile, rmtree
from tqdm import tqdm
#
# OUT_DECOYS_DIR = 'decoys'
# OUT_NATIVES_DIR = 'native'
NATIVE_SUFFIX = '_prepared_hetatms_deleted.pdb'


def combine_pdbs_into_complex(file1, file2, out_file):
    with open(file1, 'r') as f:
        in_lines1 = f.readlines()
    with open(file2, 'r') as f:
        in_lines2 = f.readlines()

    with open(out_file, 'w') as f:
        for line in in_lines1:
            f.write(line)
        for line in in_lines2:
            f.write(line)



input_native_dir = sys.argv[1] # directory with native structures
input_decoys_dir = sys.argv[2] # directory with decoys
output_dir = sys.argv[3]

if os.path.exists(output_dir):
    rmtree(output_dir)

os.makedirs(output_dir)

# out_decoys_dir = os.path.join(output_dir, OUT_DECOYS_DIR)
# out_native_dir = os.path.join(output_dir, OUT_NATIVES_DIR)

# for dir_ in [out_decoys_dir, out_native_dir]:
#     if not os.path.exists(dir_):
#         os.makedirs(dir_)

decoys_files = [f for f in os.listdir(input_decoys_dir) if f != 'log.log']
decoys_index_set = {f for f in decoys_files}

for cplx in tqdm(decoys_files, total=len(decoys_files)):
    cplx_id = cplx[:4]
    complex_dir = os.path.join(output_dir, cplx)
    complex_decoy_dir = os.path.join(complex_dir, "decoys")
    if not os.path.exists(complex_decoy_dir):
        os.makedirs(complex_decoy_dir)

    for i, decoy in enumerate(os.listdir(os.path.join(input_decoys_dir, cplx))):
        copyfile(os.path.join(input_decoys_dir, cplx, decoy),
                 os.path.join(complex_decoy_dir, f"{cplx_id}_{i+1}.pdb"))

natives_complexes = [f for f in os.listdir(input_native_dir) if f in decoys_index_set]

for cplx in tqdm(natives_complexes, total=len(natives_complexes)):
    complex_dir = os.path.join(output_dir, cplx)
    complex_native_dir = os.path.join(complex_dir, "native")
    if not os.path.exists(complex_native_dir):
        os.makedirs(complex_native_dir)
    cplx_id = cplx[:4]
    in_cplx_dir = os.path.join(input_native_dir, cplx)
    out_file = os.path.join(complex_native_dir, f"{cplx_id}.pdb")
    prepared = [os.path.join(in_cplx_dir, f) for f in os.listdir(in_cplx_dir) if f.endswith(NATIVE_SUFFIX)]
    try:
        combine_pdbs_into_complex(prepared[0], prepared[1], out_file)
    except IndexError:
        print(f"Index Error in files {prepared}, complex {cplx}")
        pass

#check decoys and native and remove if any of them
resulted_complexes = os.listdir(output_dir)

for cplx in tqdm(resulted_complexes, total=len(resulted_complexes)):
    if len(os.listdir(os.path.join(output_dir, cplx, "decoys"))) == 0 or len(os.listdir(os.path.join(output_dir, cplx, "native"))) == 0:
        rmtree(os.path.join(output_dir, cplx))
        print(f"removed dir with complex {cplx}")