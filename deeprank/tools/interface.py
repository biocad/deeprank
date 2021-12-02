import pdb2sql
import itertools
import warnings

import numpy as np



class interface(pdb2sql.interface):
    def __init__(self, pdb, **kwargs):
        super().__init__(pdb, **kwargs)

    def get_contact_residues_with_icodes(
            self,
            cutoff=8.5,
            allchains=False,
            chain1='A',
            chain2='B',
            excludeH=False,
            only_backbone_atoms=False,
            return_contact_pairs=False):

        print("Hello from get_contact_residues_with_icodes()")

        # if return_contact_pairs:

            # declare the dict
        residue_contact_pairs = {}

            # get the contact atom pairs
        atom_pairs, contact_atoms = self.get_contact_atoms (
                cutoff=cutoff,
                allchains=allchains,
                chain1=chain1,
                chain2=chain2,
                only_backbone_atoms=only_backbone_atoms,
                excludeH=excludeH,
                return_contact_pairs=True)

            # loop over the atom pair dict
        for iat1, atoms2 in atom_pairs.items():

                # get the res info of the current atom
            data1 = tuple(
                self.get(
                    'chainID,resSeq,iCode,resName',
                    rowID=[iat1])[0])

                # create a new entry in the dict if necessary
            if data1 not in residue_contact_pairs:
                residue_contact_pairs[data1] = set()

                # get the res info of the atom in the other chain
            data2 = self.get(
                'chainID,resSeq,iCode,resName', rowID=atoms2)

                # store that in the dict without double
            for resData in data2:
                residue_contact_pairs[data1].add(tuple(resData))

        for resData in residue_contact_pairs.keys():
            residue_contact_pairs[resData] = sorted(
                residue_contact_pairs[resData])


        # else:

            # get the contact atoms
        # contact_atoms = self.get_contact_atoms(
        #         cutoff=cutoff,
        #         allchains=allchains,
        #         chain1=chain1,
        #         chain2=chain2,
        #         excludeH=excludeH,
        #         only_backbone_atoms=only_backbone_atoms,
        #         return_contact_pairs=False)

            # get the residue info
        data = dict()
        residue_contact = dict()

        for chain in contact_atoms.keys():
            data[chain] = self.get(
                'chainID,resSeq,iCode,resName',
                rowID=contact_atoms[chain])
            residue_contact[chain] = sorted(
                set([tuple(resData) for resData in data[chain]]))

        return residue_contact_pairs, residue_contact

    def get_contact_atoms(
            self,
            cutoff=8.5,
            allchains=False,
            chain1='A',
            chain2='B',
            extend_to_residue=False,
            only_backbone_atoms=False,
            excludeH=False,
            return_contact_pairs=False):

        """

        Args:
            cutoff:
            allchains:
            chain1: chains of first structure
            chain2: chains of second structure
            extend_to_residue:
            only_backbone_atoms:
            excludeH:
            return_contact_pairs:

        Returns:

        """
        if allchains:
            chainIDs = self.get_chains()
        else:
            chainIDs = [chain1, chain2]

        chains = self.get_chains()
        for c in chainIDs:
            if isinstance(c, list):
                for cc in c:
                    if cc not in chains:
                        raise ValueError(
                            'chain %s not found in the structure' % cc)
            else:
                if c not in chains:
                    raise ValueError(
                        'chain %s not found in the structure' % c)

        xyz = dict()
        index = dict()
        resName = dict()
        atName = dict()

        for chains in chainIDs:
            chains_tup = tuple(chains)
            print(chains_tup)
            data = np.array(
                self.get('x,y,z,rowID,resName,name', chainID=chains))
            xyz[chains_tup] = data[:, :3].astype(float)
            index[chains_tup] = data[:, 3].astype(int)
            resName[chains_tup] = data[:, -2]
            atName[chains_tup] = data[:, -1]

        # loop through the first chain
        # TODO : loop through the smallest chain instead ...
        #index_contact_1,index_contact_2 = [],[]
        #index_contact_pairs = {}

        index_contact = dict()
        index_contact_pairs = dict()

        for chain1, chain2 in itertools.combinations(chainIDs, 2):
            ch1 = tuple(chain1)
            ch2 = tuple(chain2)
            xyz1 = xyz[ch1]
            xyz2 = xyz[ch2]

            atName1 = atName[ch1]
            atName2 = atName[ch2]
            if tuple(chain1) not in index_contact:
                index_contact[tuple(chain1)] = []

            if tuple(chain2) not in index_contact:
                index_contact[tuple(chain2)] = []

            for i, x0 in enumerate(xyz1):

                # compute the contact atoms
                contacts = np.where(
                    np.sqrt(np.sum((xyz2 - x0)**2, 1)) <= cutoff)[0]

                # exclude the H if required
                if excludeH and atName1[i][0] == 'H':
                    continue

                if len(contacts) > 0 and any(
                        [not only_backbone_atoms, atName1[i] in self.backbone_atoms]):

                    pairs = [
                        index[ch2][k] for k in contacts if any(
                            [
                                atName2[k] in self.backbone_atoms,
                                not only_backbone_atoms]) and not (
                            excludeH and atName2[k][0] == 'H')]
                    if len(pairs) > 0:
                        index_contact_pairs[index[ch1][i]] = pairs
                        index_contact[ch1] += [index[ch1][i]]
                        index_contact[ch2] += pairs

        # if no atoms were found
        if len(index_contact_pairs) == 0:
            warnings.warn('No contact atoms detected in pdb2sql')

        # get uniques
        for chain in chainIDs:
            index_contact[tuple(chain)] = sorted(set(index_contact[tuple(chain)]))

        # extend the list to entire residue
        if extend_to_residue:
            for chain in chainIDs:
                index_contact[tuple(chain)] = self._extend_contact_to_residue(
                    index_contact[tuple(chain)], only_backbone_atoms)

        # not sure that's the best way of dealing with that
        # TODO split to two functions get_contact_atoms and
        # get_contact_atom_pairs
        # if return_contact_pairs:
        #     return index_contact_pairs
        # else:
        #     return index_contact

        return index_contact_pairs, index_contact