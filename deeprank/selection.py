import numpy
from enum import Enum


class Subset:
    def __init__(self, chain_id, residue_number=None, atom_name=None, element=None):
        # None means: don't care
        self.chain = chain_id
        self.number = residue_number
        self.atom = atom_name
        self.element = element


class ProteinSelection:
    def __init__(self, chains=[], contact_distance=8.5):
        self._contact_distance = contact_distance
        self._subsets = set([])

        for chain in chains:
            self._subsets.add(Subset(chain))

    def add_subset(self, subset):
        self._subsets.add(subset)
        return self

    def contact_distance(self):
        return self._contact_distance

    @property
    def subsets(self):
        return self._subsets

    @property
    def chains(self):
        chains = set([])
        for pair in self._contact_pairs:
            chains.add(pair.chain1)
            chains.add(pair.chain2)

        for subset in self._subsets:
            chains.add(subset.chain)

        return sorted(chains)


def sql_get_contacting_atom_pairs(interface, selection, **kwargs):
    atoms = sql_get(interface, selection, 'rowID', **kwargs)

    positions = {}
    for atom, x, y, z in interface.get('rowID,x,y,z'):
        positions[atom] = (x, y, z)

    square_max_dist = numpy.square(selection.contact_distance)

    contacting = set([])
    for a1 in atoms:
        for a2 in atoms:
            if a1 != a2:
                x1, y1, z1 = positions[a1]
                x2, y2, z2 = positions[a2]
                square_dist = numpy.square(x1 - x2) + numpy.square(y1 - y2) + numpy.square(z1 - z2)

                if square_dist < square_max_dist:
                    contacting.add({a1, a2})

    return contacting

def sql_get(interface, selection, variable_name, **kwargs):

    atoms = []
    for pair in selection.contact_pairs:
        atoms.extend(interface.get('rowID', chainID=pair.chain1))
        atoms.extend(interface.get('rowID', chainID=pair.chain2))

    for subset in selection.subsets:
        selection_kwargs = {}
        if subset.chain is not None:
            selection_kwargs['chainID'] = subset.chain
        if subset.number is not None:
            selection_kwargs['resSeq'] = subset.number
        if subset.atom is not None:
            selection_kwargs['name'] = subset.atom
        if subset.element is not None:
            selection_kwargs['element'] = subset.element

        atoms.extend(interface.get('rowID', **selection_kwargs))

    return interface.get(variable_name, rowID=atoms, **kwargs)
