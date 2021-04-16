from pdb2sql import pdb2sql
from pdb2sql import StructureSimilarity
from deeprank.tools import MultiChainStructureSimilarity

decoy = pdb2sql("/home/gpu03/projects_jbremote/deeprank/complexes_data/3grw_H+L|A/decoys/3grw_5.pdb")
native = pdb2sql("/home/gpu03/projects_jbremote/deeprank/complexes_data/3grw_H+L|A/native/3grw.pdb")

sim = MultiChainStructureSimilarity(decoy, native, chains1=['H', 'L'], chains2=['A'])

irmsd = sim.compute_irmsd_pdb2sql_multi(method='svd', izone=None)
print(irmsd)