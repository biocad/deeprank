import numpy
import pdb2sql


class ProteinSelection:
    def __init__(self, atoms, center_position, contact_distance):
        self._atoms = atoms
        self._center_position = center_position
        self._contact_distance = contact_distance

    @property
    def center_position(self):
        return self._center_position

    @property
    def atoms(self):
        return self._atoms

    @property
    def contact_distance(self):
        return self._contact_distance


def get_squared_distance(p1, p2):
    return sum([numpy.square(p1[i] - p2[i]) for i in range(3)])


def get_mean_position(ps):
    s = [0] * 3
    n = len(ps)
    for p in ps:
        for i in range(3):
            s[i] += p[i]

    return [s[i] / n for i in range(3)]


def select_interface(pdb_path, chain1, chain2, atom_distance=8.5,
                     extend_to_residue=False, only_backbone=False, exclude_hydrogen=False):
    if chain1 == chain2:
        raise ValueError("cannot select an interface on twice the same chain")

    sqldb = pdb2sql.interface(pdb_path)

    try:
        atoms_per_chain = sqldb.get_contact_atoms(cutoff=atom_distance, chain1=chain1, chain2=chain2,
                                                  extend_to_residue=extend_to_residue,
                                                  only_backbone_atoms=only_backbone,
                                                  excludeH=exclude_hydrogen,
                                                  return_contact_pairs=False)

        atoms = []
        for vs in atoms_per_chain.values():
            atoms.extend(vs)
        positions = sqldb.get('x,y,z', rowID=atoms)

        return ProteinSelection(atoms, get_mean_position(positions), atom_distance)
    finally:
        sqldb._close()


def select_residue_environment(pdb_path, chain, residue_number, distance_around_residue=5.5):

    squared_max_distance = numpy.square(distance_around_residue)

    sqldb = pdb2sql.interface(pdb_path)

    try:
        center = sqldb.get('x,y,z', chainID=chain, resSeq=residue_number, name='CA')[0]

        atoms_with_positions = sqldb.get('rowID,x,y,z')

        nearby_atoms = [atom[0] for atom in atoms_with_positions if get_squared_distance(atom[1: 4], center) < squared_max_distance]

        return ProteinSelection(nearby_atoms, center, distance_around_residue)
    finally:
        sqldb._close()
