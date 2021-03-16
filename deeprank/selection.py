import numpy
import pdb2sql


class ProteinSelection:
    def __init__(self, contact_atom_pairs, center_position, max_contact_distance):
        self._contact_atom_pairs = contact_atom_pairs
        self._center_position = center_position
        self._max_contact_distance = max_contact_distance

        self._atoms = set([])
        for atom1, atom2 in self._contact_atom_pairs:
            self._atoms.add(atom1)
            self._atoms.add(atom2)

    @property
    def center_position(self):
        return self._center_position

    @property
    def contact_atom_pairs(self):
        return self._contact_atom_pairs

    @property
    def atoms(self):
        return self._atoms

    @property
    def max_contact_distance(self):
        return self._max_contact_distance


def get_squared_distance(p1, p2):
    return sum([numpy.square(p1[i] - p2[i]) for i in range(3)])


def get_mean_position(ps):
    s = [0] * 3
    n = len(ps)
    for p in ps:
        for i in range(3):
            s[i] += p[i]

    return [s[i] / n for i in range(3)]


def select_interface(pdb_path, chain1, chain2, max_atom_distance=8.5,
                     extend_to_residue=False, only_backbone=False, exclude_hydrogen=False):
    if chain1 == chain2:
        raise ValueError("cannot select an intra chain interface")

    sqldb = pdb2sql.interface(pdb_path)

    try:
        d = sqldb.get_contact_atoms(cutoff=max_atom_distance, chain1=chain1, chain2=chain2,
                                    extend_to_residue=extend_to_residue,
                                    only_backbone_atoms=only_backbone,
                                    excludeH=exclude_hydrogen,
                                    return_contact_pairs=True)

        pairs = []
        atoms = set([])
        for atom1, atoms2 in d.items():
            atoms.add(atom1)
            for atom2 in atoms2:
                atoms.add(atom2)
                pairs.append((atom1, atom2))

        positions = [sqldb.get('x,y,z', rowID=atom)[0] for atom in atoms]

        return ProteinSelection(pairs, get_mean_position(positions), max_atom_distance)
    finally:
        sqldb._close()


def select_residue_environment(pdb_path, chain, residue_number, distance_around_residue=5.5):

    squared_max_distance = numpy.square(distance_around_residue)

    sqldb = pdb2sql.interface(pdb_path)

    try:
        center_atom = sqldb.get('rowID', chainID=chain, resSeq=residue_number, name='CA')[0]
        center = sqldb.get('x,y,z', rowID=center_atom)[0]

        atoms_with_positions = sqldb.get('rowID,x,y,z')

        nearby_atoms = [atom[0] for atom in atoms_with_positions if get_squared_distance(atom[1: 4], center) < squared_max_distance]

        pairs = [(center_atom, nearby_atom) for nearby_atom in nearby_atoms if nearby_atom != center_atom]

        return ProteinSelection(pairs, center, distance_around_residue)
    finally:
        sqldb._close()
