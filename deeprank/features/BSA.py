import warnings
import itertools

import pdb2sql

from deeprank.features import FeatureClass
from deeprank.tools.interface import interface

try:
    import freesasa

except ImportError:
    warnings.warn('freesasa module not found')
    raise


class BSA(FeatureClass):

    def __init__(self, pdb_data, interface, precomputed_dict, chain1='A', chain2='B'):
        """Compute the burried surface area feature.

        Freesasa is required for this feature.
        From Freesasa version 2.0.3 the Python bindings are released
        as a separate module. They can be installed using
        >>> pip install freesasa

        Args:
            pdb_data (list(byte) or str): pdb data or pdb filename
            chain1 (str, optional): name of the first chain
            chain2 (str, optional): name of the second chain

        Example:
            >>> bsa = BSA('1AK4.pdb')
            >>> bsa.get_structure()
            >>> bsa.get_contact_residue_sasa()
            >>> bsa.sql._close()
        """
        self.pdb_data = pdb_data
        self.sql = interface
        self.contacts = precomputed_dict['contacts']
        # self.sql = interface(pdb_data)
        self.chain1 = chain1
        self.chain2 = chain2
        # now each of chain1 and chain2 is list o tuple
        self.chains_label = [tuple(chain1), tuple(chain2)]

        self.feature_data = {}
        self.feature_data_xyz = {}

        freesasa.setVerbosity(freesasa.nowarnings)

    def get_structure(self):
        """Get the pdb structure of the molecule."""

        # we can have a str or a list of bytes as input
        if isinstance(self.pdb_data, str):
            self.complex = freesasa.Structure(self.pdb_data)
        else:
            self.complex = freesasa.Structure()
            atomdata = self.sql.get('name,resName,resSeq,chainID,x,y,z')
            for atomName, residueName, residueNumber, chainLabel, x, y, z \
                    in atomdata:
                atomName = '{:>2}'.format(atomName[0])
                self.complex.addAtom(atomName, residueName,
                                     residueNumber, chainLabel, x, y, z)
        self.result_complex = freesasa.calc(self.complex)

        self.chains = {}
        self.result_chains = {}
        for label in self.chains_label:
            self.chains[label] = freesasa.Structure()
            atomdata = self.sql.get(
                'name,resName,resSeq,chainID,x,y,z', chainID=list(label))
            for atomName, residueName, residueNumber, chainLabel, x, y, z \
                    in atomdata:
                atomName = '{:>2}'.format(atomName[0])
                self.chains[label].addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z)
            self.result_chains[label] = freesasa.calc(self.chains[label])

    def get_contact_residue_sasa(self, cutoff=5.5):
        """Compute the feature of BSA.

            It generates following feature:
                bsa

        Raises:
            ValueError: No interface residues found.
        """

        self.bsa_data = {}
        self.bsa_data_xyz = {}

        # ctc_res_list = []
        # for ch1, ch2 in itertools.product(self.chain1, self.chain2):
        #     ctc_res = self.sql.get_contact_residues_with_icodes(cutoff=cutoff, chain1=ch1, chain2=ch2)
        #     ctc_res_list = list(set(ctc_res_list + ctc_res[ch1] + ctc_res[ch2]))
        print(f"Cutoff in BSA call = {cutoff}")
        # ctc_res = self.sql.get_contact_residues_with_icodes(cutoff=cutoff, chain1=self.chain1, chain2=self.chain2)
        ctc_res = self.contacts
        ctc_res_list = ctc_res[tuple(self.chain1)] + ctc_res[tuple(self.chain2)]

        # ctc_res_all = self.sql.get_contact_residues_with_icodes(cutoff=cutoff, chain1=self.chain1, chain2=self.chain2)
        # ctc_res = []
        # for c in self.chain1:
        #     ctc_res += ctc_res_all[c]
        #
        # for c in self.chain2:
        #     ctc_res += ctc_res_all[c]

        # ctc_res = self.sql.get_contact_residues(cutoff=cutoff, chain1=self.chain1, chain2=self.chain2)
        # ctc_res = ctc_res[self.chain1] + ctc_res[self.chain2]

        # handle with small interface or no interface
        total_res = len(ctc_res_list)
        if total_res == 0:
            raise ValueError(
                f"No interface residue found with the cutoff {cutoff}Å."
                f" Failed to calculate the feature BSA")
        elif total_res < 5:  # this is an empirical value
            warnings.warn(
                f"Only {total_res} interface residues found with cutoff"
                f" {cutoff}Å. Be careful with using the feature BSA")

        for res in ctc_res_list:

            # define the selection string and the bsa for the complex
            select_str = ('res, (resi %d) and (chain %s)' % (res[1], res[0]),)
            asa_complex = freesasa.selectArea(
                select_str, self.complex, self.result_complex)['res']

            # define the selection string and the bsa for the isolated
            select_str = ('res, resi %d' % res[1],)

            chains_key = self.chains_label[0] if res[0] in self.chains_label[0] else self.chains_label[1]
            asa_unbound = freesasa.selectArea(
                select_str, self.chains[chains_key],
                self.result_chains[chains_key])['res']

            # define the bsa
            bsa = asa_unbound - asa_complex

            # define the xyz key: (chain,x,y,z)
            chain = 0 if res[0] in self.chain1 else 1
            # chain = {self.chain1: 0, self.chain2: 1}[res[0]]

            # get the center
            _, xyz = self.get_residue_center(self.sql, res=res)
            xyzkey = tuple([chain] + xyz[0])

            # put the data in dict
            self.bsa_data[res] = [bsa]
            self.bsa_data_xyz[xyzkey] = [bsa]

        # pyt the data in dict
        self.feature_data['bsa'] = self.bsa_data
        self.feature_data_xyz['bsa'] = self.bsa_data_xyz

########################################################################
#
#   THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
########################################################################


def __compute_feature__(pdb_data, featgrp, featgrp_raw, chain1, chain2):
    """Main function called in deeprank for the feature calculations.

    Args:
        pdb_data (list(bytes)): pdb information
        featgrp (str): name of the group where to save xyz-val data
        featgrp_raw (str): name of the group where to save human readable data
        chain1 (str): First chain ID
        chain2 (str): Second chain ID
    """

    # create the BSA instance
    bsa = BSA(pdb_data, chain1, chain2)

    # get the structure/calc
    bsa.get_structure()

    # get the feature info
    bsa.get_contact_residue_sasa()

    # export in the hdf5 file
    bsa.export_dataxyz_hdf5(featgrp)
    bsa.export_data_hdf5(featgrp_raw)

    # close the file
    bsa.sql._close()


def __compute_feature_ram__(pdb_data, featgrp, featgrp_raw, chain1, chain2, interface, precomputed_dict):
    """Main function called in deeprank for the feature calculations.

    Args:
        pdb_data (list(bytes)): pdb information
        featgrp (str): name of the group where to save xyz-val data
        featgrp_raw (str): name of the group where to save human readable data
        chain1 (str): First chain ID
        chain2 (str): Second chain ID
    """

    # create the BSA instance
    bsa = BSA(pdb_data, interface, precomputed_dict, chain1, chain2)

    # get the structure/calc
    bsa.get_structure()

    # get the feature info
    bsa.get_contact_residue_sasa()

    # export in the hdf5 file
    # bsa.export_dataxyz_hdf5(featgrp)
    # bsa.export_data_hdf5(featgrp_raw)

    # close the file
    # bsa.sql._close()
    return bsa.feature_data, bsa.feature_data_xyz


########################################################################
#
#       TEST THE CLASS
#
########################################################################

if __name__ == '__main__':

    import os
    from pprint import pprint
    # get base path */deeprank, i.e. the path of deeprank package
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))))
    pdb_file = os.path.join(base_path, "test/1AK4/native/1AK4.pdb")

    bsa = BSA(pdb_file)
    bsa.get_structure()
    bsa.get_contact_residue_sasa()
    bsa.sql._close()

    pprint(bsa.feature_data)
    print()
    pprint(bsa.feature_data_xyz)
