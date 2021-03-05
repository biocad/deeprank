import os
import warnings

import numpy as np
import pdb2sql

from deeprank.features import FeatureClass
from deeprank.selection import ProteinSelection, sql_get


class AtomicFeature(FeatureClass):

    def __init__(self, pdbfile, selection=ProteinSelection(chains=['A', 'B']), param_charge=None,
                param_vdw=None, patch_file=None,
                verbose=False):
        """Compute the Coulomb, van der Waals interaction and charges.

        Args:

            pdbfile (str): pdb file of the molecule

            selection (selection object): the protein region of interest

            param_charge (str): file name of the force field file
                containing the charges e.g. protein-allhdg5.4_new.top.
                Must be of the format:
                * CYM  atom O   type=O      charge=-0.500 end
                * ALA    atom N   type=NH1     charge=-0.570 end

            param_vdw (str): file name of the force field containing
                vdw parameters e.g. protein-allhdg5.4_new.param.
                Must be of the format:
                * NONBonded  CYAA    0.105   3.750       0.013    3.750
                * NONBonded  CCIS    0.105   3.750       0.013    3.750

            patch_file (str): file name of a valid patch file for
                the parameters e.g. patch.top.
                The way we handle the patching is very manual and
                should be made more automatic.

            verbose (bool): print or not.

        Examples:
            >>> pdb = '1AK4_100w.pdb'
            >>>
            >>> # get the force field included in deeprank
            >>> # if another FF has been used to compute the ref
            >>> # change also this path to the correct one
            >>> FF = pkg_resources.resource_filename(
            >>>     'deeprank.features','') + '/forcefield/'
            >>>
            >>> # declare the feature calculator instance
            >>> atfeat = AtomicFeature(pdb,
            >>>    param_charge = FF + 'protein-allhdg5-4_new.top',
            >>>    param_vdw    = FF + 'protein-allhdg5-4_new.param',
            >>>    patch_file   = FF + 'patch.top')
            >>>
            >>> # assign parameters
            >>> atfeat.assign_parameters()
            >>>
            >>> # only compute the pair interactions here
            >>> atfeat.evaluate_pair_interaction(save_interactions=test_name)
            >>>
            >>> # close the db
            >>> atfeat.sqldb._close()
        """

        super().__init__("Atomic")

        # set a few things
        self.pdbfile = pdbfile
        self.selection = selection
        self.param_charge = param_charge
        self.param_vdw = param_vdw
        self.patch_file = patch_file
        self.verbose = verbose

        # a few constant
        self.eps0 = 1
        self.c = 332.0636
        self.residue_key = 'chainID, resSeq, resName'
        self.atom_key = 'chainID, resSeq, resName, name'

        # read the pdb as an sql
        self.sqldb = pdb2sql.pdb2sql(self.pdbfile)

        # read the force field
        self.read_charge_file()

        if patch_file is not None:
            self.read_patch()
        else:
            self.patch_charge, self.patch_type = {}, {}

        # read the vdw param file
        self.read_vdw_file()

        # get the atoms
        self.selection_atoms = sql_get(self.sqldb, self.selection, "rowID")
        self.selection_chains = sorted(set(sql_get(self.sqldb, self.selection, "chainID")))

    ####################################################################
    #
    #   READ INPUT FILES
    #
    ####################################################################

    def read_charge_file(self):
        """Read the .top file given in entry.

        This function creates:

        - self.charge: dictionary  {(resname,atname):charge}
        - self.valid_resnames: list ['VAL','ALP', .....]
        - self.at_name_type_convertor: dict {(resname,atname):attype}
        """

        with open(self.param_charge) as f:
            data = f.readlines()

        # loop over all the data
        self.charge = {}
        self.at_name_type_convertor = {}
        resnames = []

        # loop over the file
        for l in data:

            # split the line
            words = l.split()

            # get the resname/atname
            res, atname = words[0], words[2]

            # get the charge
            ind = l.find('charge=')
            q = float(l[ind + 7:ind + 13])

            # get the type
            attype = words[3].split('=')[-1]

            # store the charge
            self.charge[(res, atname)] = q

            # put the resname in a list so far
            resnames.append(res)

            # dictionary for conversion name/type
            self.at_name_type_convertor[(res, atname)] = attype

        self.valid_resnames = list(set(resnames))

    def read_patch(self):
        """Read the patchfile.

        This function creates

            - self.patch_charge: Dict {(resName,atName): charge}
            - self.patch_type : Dict {(resName,atName): type}
        """

        with open(self.patch_file) as f:
            data = f.readlines()

        self.patch_charge, self.patch_type = {}, {}

        for l in data:
            # ignore comments
            if l[0] != '#' and l[0] != '!' and len(l.split()) > 0:
                words = l.split()

                # get the new charge
                ind = l.find('CHARGE=')
                q = float(l[ind + 7:ind + 13])
                self.patch_charge[(words[0], words[3])] = q

                # get the new type if any
                ind = l.find('TYPE=')
                if ind != -1:
                    type_ = l[ind + 5:ind + 9]
                    self.patch_type[(words[0], words[3])] = type_.strip()

    def read_vdw_file(self):
        """Read the .param file.

        The param file must be of the form:

            NONBONDED ATNAME 0.10000 3.298765 0.100000 3.089222

            - First two numbers are for inter-chain interations
            - Last two nmbers are for intra-chain interactions
              (We only compute the interchain here)

        This function creates

            - self.vdw: dictionary {attype:[E1,S1]}
        """

        with open(self.param_vdw) as f:
            data = f.readlines()

        self.vdw_param = {}

        for line in data:
            # split the atom
            line = line.split()

            # empty line
            if len(line) == 0:
                continue

            # comment
            if line[0][0] == '#':
                continue

            self.vdw_param[line[1]] = list(map(float, line[2:4]))

    def _extend_selection_to_residue(self):
        """Extend the contact atoms to entire residue where one atom is
        contacting."""

        # extract the data
        data = self.sqldb.get(self.residue_key, self.selection_atoms)

        # extract uniques
        res = list(set(data))

        # extend
        index = []
        for resdata in res:
            chainID, resSeq, resName = resdata
            index += self.sqldb.get('rowID', chainID=chainID,
                                             resName=resName,
                                             resSeq=resSeq)

        return index

    ####################################################################
    #
    #   Assign parameters
    #
    ####################################################################

    def assign_parameters(self):
        """Assign to each atom in the pdb its charge and vdw interchain
        parameters.

        Directly deals with the patch so that we don't loop over the
        residues multiple times.
        """

        # get all the resnumbers
        if self.verbose:
            print('-- Assign force field parameters')

        data = self.sqldb.get(self.residue_key)
        natom = len(data)
        data = np.unique(np.array(data), axis=0)

        # declare the parameters for future insertion in SQL
        atcharge = np.zeros(natom)
        ateps = np.zeros(natom)
        atsig = np.zeros(natom)

        # check
        attype = np.zeros(natom, dtype='<U5')
        ataltResName = np.zeros(natom, dtype='<U5')

        # loop over all the residues
        for chain, resNum, resName in data:

            # atom types of the residue
            #query = "WHERE chainID='%s' AND resSeq=%s" %(chain,resNum)
            atNames = np.array(self.sqldb.get(
                'name', chainID=chain, resSeq=resNum))
            rowID = np.array(self.sqldb.get(
                'rowID', chainID=chain, resSeq=resNum))

            # get the alternative resname
            altResName = self._get_altResName(resName, atNames)

            # get the charge of this residue
            atcharge[rowID] = self._get_charge(resName, altResName, atNames)

            # get the vdw parameters
            eps, sigma, type_ = self._get_vdw(resName, altResName, atNames)
            ateps[rowID] += eps
            atsig[rowID] += sigma

            ataltResName[rowID] = altResName
            attype[rowID] = type_

        # put the charge in SQL
        self.sqldb.add_column('CHARGE')
        self.sqldb.update_column('CHARGE', atcharge)

        # put the VDW in SQL
        self.sqldb.add_column('eps')
        self.sqldb.update_column('eps', ateps)

        self.sqldb.add_column('sig')
        self.sqldb.update_column('sig', atsig)

        self.sqldb.add_column('type', 'TEXT')
        self.sqldb.update_column('type', attype)

        self.sqldb.add_column('altRes', 'TEXT')
        self.sqldb.update_column('altRes', ataltResName)

    @staticmethod
    def _get_altResName(resName, atNames):
        """Apply the patch data.

        This is adopted from preScan.pl
        This is very static and I don't quite like it
        The structure of the dictionary is as following

        { NEWRESTYPE: 'OLDRESTYPE',
                       [atom types that must be present],
                       [atom types that must NOT be present]]}

        Args:
            resName (str): name of the residue
            atNames (list(str)): names of the atoms
        """

        new_type = {
            'PROP': ['all', ['HT1', 'HT2'], []],
            'NTER': ['all', ['HT1', 'HT2', 'HT3'], []],
            'CTER': ['all', ['OXT'], []],
            'CTN': ['all', ['NT', 'HT1', 'HT2'], []],
            'CYNH': ['CYS', ['1SG'], ['2SG']],
            'DISU': ['CYS', ['1SG', '2SG'], []],
            'HISE': ['HIS', ['ND1', 'CE1', 'CD2', 'NE2', 'HE2'], ['HD1']],
            'HISD': ['HIS', ['ND1', 'CE1', 'CD2', 'NE2', 'HD1'], ['HE2']]
        }

        # this works fine now

        altResName = resName
        for key, values in new_type.items():
            res, atpres, atabs = values
            if res == resName or res == 'all':
                if all(x in atNames for x in atpres) and all(
                        x not in atNames for x in atabs):
                    altResName = key

        return altResName

    def _get_vdw(self, resName, altResName, atNames):
        """Get vdw itneraction terms.

        Args:
            resName (str): name of the residue
            altResName (str): alternative name of the residue
            atNames (list(str)): names of the atoms
        """

        # in case the resname is not valid
        if resName not in self.valid_resnames:
            vdw_eps = [0.00] * len(atNames)
            vdw_sigma = [0.00] * len(atNames)
            type_ = ['None'] * len(atNames)

            return vdw_eps, vdw_sigma, type_

        vdw_eps, vdw_sigma, type_ = [], [], []

        for at in atNames:

            if (altResName, at) in self.patch_type:
                type_.append(self.patch_type[(altResName, at)])
                vdw_data = self.vdw_param[self.patch_type[(altResName, at)]]
                vdw_eps.append(vdw_data[0])
                vdw_sigma.append(vdw_data[1])

            elif (resName, at) in self.at_name_type_convertor:
                type_.append(self.at_name_type_convertor[(resName, at)])
                vdw_data = self.vdw_param[self.at_name_type_convertor[(
                    resName, at)]]
                vdw_eps.append(vdw_data[0])
                vdw_sigma.append(vdw_data[1])

            else:
                type_.append('None')
                vdw_eps.append(0.0)
                vdw_sigma.append(0.0)
                warnings.warn(f"Atom type {at} not found for "
                              f"resType {resName} or patch type {altResName}. "
                              f"Set vdw eps and sigma to 0.0.")

        return vdw_eps, vdw_sigma, type_

    def _get_charge(self, resName, altResName, atNames):
        """Get the charge information.

        Args:
            resName (str): name of the residue
            altResName (str): alternative name of the residue
            atNames (list(str)): names of the atoms
        """
        # in case the resname is not valid
        if resName not in self.valid_resnames:
            q = [0.0] * len(atNames)
            return q

        # assign the charges
        q = []
        for at in atNames:
            if (altResName, at) in self.patch_charge:
                q.append(self.patch_charge[(altResName, at)])
            elif (resName, at) in self.charge:
                q.append(self.charge[(resName, at)])
            else:
                q.append(0.0)
                warnings.warn(f"Atom type {at} not found for "
                              f"resType {resName} or patch type {altResName}. "
                              f"Set charge to 0.0.")
        return q

    ####################################################################
    #
    #   Simple charges
    #
    ####################################################################

    def evaluate_charges(self, extend_contact_to_residue=False):
        """Evaluate the charges.

        Args:
            extend_contact_to_residue (bool, optional): extend to res
        """
        if self.verbose:
            print('-- Compute list charge for contact atoms only')

        # extract information from the pdb2sq
        xyz = np.array(self.sqldb.get('x,y,z'))
        atinfo = self.sqldb.get(self.atom_key)

        charge = np.array(self.sqldb.get('CHARGE'))

        # define the dictionaries
        charge_data = {}
        charge_data_xyz = {}

        # entire residue or not
        if extend_contact_to_residue:
            index_atoms = self._extend_selection_to_residue()
        else:
            index_atoms = self.selection_atoms

        # loop over the chain A
        for i in atoms:

            # atinfo
            key = tuple(atinfo[i])

            # store in the dicts
            charge_data[key] = [charge[i]]

            # xyz format
            chain_dict = [self.selection_chains.index(key[0])]
            key = tuple(chain_dict + xyz[i, :].tolist())
            charge_data_xyz[key] = [charge[i]]

        # add the electrosatic feature
        self.feature_data['charge'] = charge_data
        self.feature_data_xyz['charge'] = charge_data_xyz

    ####################################################################
    #
    #   PAIR INTERACTIONS
    #
    ####################################################################

    def evaluate_pair_interaction(self, print_interactions=False,
                                  save_interactions=False):
        """Evalaute the pair interactions (coulomb and vdw).

        Args:
            print_interactions (bool, optional): print data to screen
            save_interactions (bool, optional): save the itneractions to file.
        """

        if self.verbose:
            print('-- Compute interaction energy for contact pairs only')

        # extract information from the pdb2sql
        contacting_pairs = sql_get_contacting_atom_pairs(self.sqldb, self.selection)
        xyz = self.sqldb.get('x,y,z')
        atinfo = self.sqldb.get(self.atom_key)

        charge = np.array(self.sqldb.get('CHARGE'))
        vdw = np.array(self.sqldb.get('eps,sig'))
        eps, sig = vdw[:, 0], vdw[:, 1]

        # define the dictionaries
        # these holds data like
        # chainID resname resSeq,name values
        electro_data = {}
        vdw_data = {}

        # define the dict that hold
        #  x y z values
        electro_data_xyz = {}
        vdw_data_xyz = {}

        # handle the export of the interaction breakdown
        _save_ = False
        if save_interactions:
            if save_interactions:
                save_interactions = './'
            if os.path.isdir(save_interactions):
                fname = os.path.join(save_interactions,
                                     'atomic_pair_interaction.dat')
            else:
                fname = save_interactions
            f = open(fname, 'w')
            _save_ = True

        # total energy terms
        ec_tot, evdw_tot = 0, 0

        # loop over the contacts
        for atomA, atomB in contacting_pairs:

            # coulomb terms
            r = np.sqrt(np.sum((xyz[atomB] - xyz[atomA])**2, 1))
            if r == 0:
                r = 3.0

            q1q2 = charge[atomA] * charge[atomB]
            ec = q1q2 * self.c / (self.eps0 * r) * (1 - (r / self.selection.contact_distance)**2) ** 2

            # coulomb terms
            sigma_avg = 0.5 * (sig[atomA] + sig[atomB])
            eps_avg = np.sqrt(eps[atomA] * eps[atomB])

            # normal LJ potential
            evdw = 4.0 * eps_avg * \
                ((sigma_avg / r)**12 - (sigma_avg / r)**6) * self._prefactor_vdw(r)

            # total energy terms
            ec_tot += ec
            evdw_tot += evdw

            for atom in [atomA, atomB]:

                # store in the dicts
                atom_key = tuple(atinfo[atom])
                electro_data[atom_key] = electro_data.get(atom_key, 0) + ec
                vdw_data[atom_key] = vdw_data.get(atom_key, 0) + evdw

                # store in the xyz dict
                xyz_key = tuple([atom] + xyz[atom])
                electro_data_xyz[xyz_key] = electro_data_xyz.get(xyz_key, 0) + ec
                vdw_data_xyz[xyz_key] = vdw_data_xyz.get(xyz_key, 0) + evdw

                # print the result
                if _save_ or print_interactions:

                    line = ''

                    for atom in [atomA, atomB]:
                        atom_key = tuple(atinfo[atom])

                        line += '{:<3s}'.format(atom_key[0])
                        line += '\t{:>1d}'.format(atom_key[1])
                        line += '\t{:>4s}'.format(atom_key[2])
                        line += '\t{:^4s}'.format(atom_key[3])

                    line += '\t{: 6.3f}'.format(r)
                    line += '\t{: f}'.format(ec)
                    line += '\t{: e}'.format(evdw)

                    # print and/or save the interactions
                    if print_interactions:
                        print(line)

                    if _save_:
                        line += '\n'
                        f.write(line)

        # print the total interactions
        if print_interactions or _save_:
            line = '\n\n'
            line += 'Total Evdw  = {:> 12.8f}\n'.format(evdw_tot)
            line += 'Total Eelec = {:> 12.8f}\n'.format(ec_tot)
            if print_interactions:
                print(line)
            if _save_:
                f.write(line)

        # close export file
        if _save_:
            f.close()
            print(f'AtomicFeature coulomb and vdw exported to file {fname}')

        # add the electrosatic feature
        self.feature_data['coulomb'] = electro_data
        self.feature_data_xyz['coulomb'] = electro_data_xyz

        # add the vdw feature
        self.feature_data['vdwaals'] = vdw_data
        self.feature_data_xyz['vdwaals'] = vdw_data_xyz


    @staticmethod
    def _prefactor_vdw(r):
        """prefactor for vdw interactions."""

        r_off, r_on = 8.5, 6.5
        r2 = r**2
        pref = (r_off**2 - r2)**2 * (r_off**2 - r2 - 3 *
                                     (r_on**2 - r2)) / (r_off**2 - r_on**2)**3
        pref[r > r_off] = 0.
        pref[r < r_on] = 1.0
        return pref


########################################################################
#
#   THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
########################################################################

def __compute_feature__(pdb_data, featgrp, featgrp_raw, selection):
    """Main function called in deeprank for the feature calculations.

    Args:
        pdb_data (list(bytes)): pdb information
        featgrp (str): name of the group where to save xyz-val data
        featgrp_raw (str): name of the group where to save human readable data
        chain1 (str): First chain ID
        chain2 (str): Second chain ID
    """
    path = os.path.dirname(os.path.realpath(__file__))
    FF = path + '/forcefield/'

    atfeat = AtomicFeature(pdb_data,
                           selection=selection,
                           param_charge=FF + 'protein-allhdg5-4_new.top',
                           param_vdw=FF + 'protein-allhdg5-4_new.param',
                           patch_file=FF + 'patch.top')

    atfeat.assign_parameters()

    # only compute the pair interactions here
    atfeat.evaluate_pair_interaction(print_interactions=False)

    # compute the charges
    # here we extand the contact atoms to
    # entire residue containing at least 1 contact atom
    atfeat.evaluate_charges(extend_contact_to_residue=True)

    # export in the hdf5 file
    atfeat.export_dataxyz_hdf5(featgrp)
    atfeat.export_data_hdf5(featgrp_raw)

    # close
    atfeat.sqldb._close()


########################################################################
#
#  TEST THE CLASS
#
########################################################################

if __name__ == '__main__':

    from pprint import pprint
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))))
    pdb_file = os.path.join(base_path, "test/1AK4/native/1AK4.pdb")
    FF = os.path.join(base_path, 'deeprank/features/forcefield/')

    atfeat = AtomicFeature(pdb_file,
                           param_charge=FF + 'protein-allhdg5-4_new.top',
                           param_vdw=FF + 'protein-allhdg5-4_new.param',
                           patch_file=FF + 'patch.top',
                           verbose=True)

    atfeat.assign_parameters()
    atfeat.evaluate_pair_interaction()
    atfeat.evaluate_charges(extend_contact_to_residue=True)
    atfeat.sqldb._close()

    # export in the hdf5 file
    pprint(atfeat.feature_data)
    print()
    pprint(atfeat.feature_data_xyz)
