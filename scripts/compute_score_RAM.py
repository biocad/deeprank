import os
from deeprank.generate import DataGeneratorRAM

PATH_TO_COMPLEX = '/home/gpu03/projects_jbremote/deeprank/test_apply_deeprank_model/1adq_H+L|A/'

chains1 = ['H', 'L']
chains2 = ['A']

complex_data = DataGeneratorRAM(pdb_source=PATH_TO_COMPLEX,
                                pssm_source='.',
                                compute_features=['deeprank.features.AtomicFeature'],
                                                       # 'deeprank.features.ResidueDensity',
                                                       # 'deeprank.features.BSA'],
                                compute_targets=['deeprank.targets.identity_target'],
                                chain1=chains1,
                                chain2=chains2,
                                hdf5=PATH_TO_COMPLEX)

complex_data.create_database()

print("Database created successfully!")