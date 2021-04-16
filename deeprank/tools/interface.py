import pdb2sql


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

        if return_contact_pairs:

            # declare the dict
            residue_contact_pairs = {}

            # get the contact atom pairs
            atom_pairs = self.get_contact_atoms (
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

            return residue_contact_pairs

        else:

            # get the contact atoms
            contact_atoms = self.get_contact_atoms(
                cutoff=cutoff,
                allchains=allchains,
                chain1=chain1,
                chain2=chain2,
                excludeH=excludeH,
                only_backbone_atoms=only_backbone_atoms,
                return_contact_pairs=False)

            # get the residue info
            data = dict()
            residue_contact = dict()

            for chain in contact_atoms.keys():
                data[chain] = self.get(
                    'chainID,resSeq,iCode,resName',
                    rowID=contact_atoms[chain])
                residue_contact[chain] = sorted(
                    set([tuple(resData) for resData in data[chain]]))

            return residue_contact