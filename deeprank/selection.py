from enum import Enum


class ContactPair:
    def __init__(self, chain1, chain2, distance=8.5):
        self.chain1 = chain1
        self.chain2 = chain2
        self.distance = distance


class Subset:
    def __init__(self, chain_id, residue_number=None, atom_name=None):
        # None means: don't care
        self.chain = chain_id
        self.number = residue_number
        self.atom = atom_name


class ProteinSelection:
    def __init__(self):
        self._subsets = set([])
        self._contact_pairs = set([])

    def add_subset(self, subset):
        self._subsets.add(subset)
        return self

    def add_contact_pair(self, pair):
        self._contact_pairs.add(pair)
        return self

    @property
    def contact_pairs(self):
        return self._contact_pairs

    @property
    def subsets(self):
        return self._subsets


def sql_get(interface, selection, variable_name, **kwargs):

    atoms = []
    for pair in selection.contact_pairs:
        atoms.extend(interface.get_contact_atoms(chain1=pair.chain1, chain2=pair.chain2, cutoff=pair.distance))

    for subset in selection.subsets:
        selection_kwargs = {}
        if subset.chain is not None:
            selection_kwargs['chainID'] = subset.chain
        if subset.number is not None:
            selection_kwargs['resSeq'] = subset.number
        if subset.atom is not None:
            selection_kwargs['name'] = subset.atom

        atoms.extend(interface.get('rowID', **selection_kwargs))

    return interface.get(variable_name, rowID=atoms, **kwargs)
