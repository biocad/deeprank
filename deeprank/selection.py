import logging
import numpy
import pdb2sql

_log = logging.getLogger(__name__)


class ProteinSelection:
    residue_key = "chainID,resSeq,resName"


class InterfaceSelection(ProteinSelection):
    def __init__(self, chain1, chain2):
        self.chain1 = chain1
        self.chain2 = chain2


class MutantSelection(ProteinSelection):
    def __init__(self, chain, residue_number):
        self.chain = chain
        self.residue_number = residue_number


def get_contact_atoms(sqldb, selection, max_interatomic_distance):
    if type(selection) == InterfaceSelection:

        d = sqldb.get_contact_atoms(cutoff=max_interatomic_distance,
                                    chain1=selection.chain1, chain2=selection.chain2,
                                    return_contact_pairs=True)
        pairs = set([])
        for a1, as2 in d.items():
            for a2 in as2:
                pairs.add(tuple(sorted([a1, a2])))

        return sorted(pairs)

    elif type(selection) == MutantSelection:
        return get_atoms_around_residue(selection.chain, selection.residue_numbers,
                                        max_interatomic_distance)
    else:
        raise TypeError(type(selection))


def get_average_position(sqldb, atom_numbers):
    ps = sqldb.get('x,y,z', rowID=atom_numbers)

    s = [0] * 3
    n = len(ps)
    for p in ps:
        for i in range(3):
            s[i] += p[i]

    return [s[i] / n for i in range(3)]


def get_atoms(sqldb, selection, max_interatomic_distance):
    atoms = set([])
    for atom1, atom2 in get_contact_atoms(sqldb, selection, max_interatomic_distance):
        atoms.add(atom1)
        atoms.add(atom2)

    return sorted(atoms)


def get_center(sqldb, selection, max_interatomic_distance):
    if type(selection) == InterfaceSelection:
        return get_average_position(sqldb, get_atoms(sqldb, selection, max_interatomic_distance))

    elif type(selection) == MutantSelection:
        return get_residue_position(sqldb, selection.chain, selection.residue_number)

    else:
        raise TypeError(type(selection))


def get_contact_residues(sqldb, selection, max_interatomic_distance):
    if type(selection) == InterfaceSelection:
        d = sqldb.get_contact_residues(cutoff=max_interatomic_distance,
                                       chain1=selection.chain1, chain2=selection.chain2,
                                       return_contact_pairs=True)
        pairs = set()
        for r1, rs2 in d.items():
            for r2 in rs2:
                pairs.add(tuple(sorted([tuple(r1), tuple(r2)])))

        return sorted(pairs)

    elif type(selection) == MutantSelection:
        return get_residues_around_residue(selection.chain, selection.residue_number, max_interatomic_distance)

    else:
        raise TypeError(type(selection))

def get_residues(sqldb, selection, max_interatomic_distance):
    residues = set([])
    if type(selection) == MutantSelection:
        residues.add(sqldb.get(ProteinSelection.residue_key, chainID=selection.chain, resSeq=selection.residue_number)[0])

    for res1, res2 in get_contact_residues(sqldb, selection, max_interatomic_distance):
        residues.add(res1)
        residues.add(res2)

    return sorted(residues)


def get_chains(sqldb, selection, max_interatomic_distance):
    chains = set([])
    if type(selection) == InterfaceSelection:
        chains.add(selection.chain1)
        chains.add(selection.chain2)

    elif type(selection) == MutantSelection:
        chains.add(selection.chain)

    for res1, res2 in get_contact_residues(sqldb, selection, max_interatomic_distance):
        for res in [res1, res2]:
            keys = ProteinSelection.residue_key.split(',')

            d = {keys[i]: res[i] for i in range(len(keys))}
            chains.add(d['chainID'])

    return sorted(chains)


def get_squared_distance(p1, p2):
    return sum([numpy.square(p1[i] - p2[i]) for i in range(3)])

