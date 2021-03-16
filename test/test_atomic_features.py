import logging
import unittest
from nose.tools import eq_

import numpy as np
import pkg_resources

from deeprank.selection import select_interface
from deeprank.features import AtomicFeature


_log = logging.getLogger(__name__)


def _close(what, value1, value2, **kwargs):
    if not np.isclose(value1, value2, **kwargs):
        raise AssertionError("{} not close: {} vs. {}".format(what, value1, value2))


def _make_atom(res_name, chain, res_num, atom_name, atom_key):

    info = []
    for index, key in enumerate(atom_key.split(',')):
        key = key.strip()
        if key == 'name':
            info.append(atom_name)

        elif key == 'chainID':
            info.append(chain)

        elif key == 'resSeq':
            info.append(res_num)

        elif key == 'resName':
            info.append(res_name)

    return tuple(info)


def _parse_ref(file_, atom_key):

    pairs = []
    total_vdw_energy = None
    total_elec_energy = None

    for line in file_:
        if len(line.strip()) == 0 or line.startswith('#'):
            continue

        elif line.startswith("Total Evdw = "):
            total_vdw_energy = float(line.split('=')[1])

        elif line.startswith("Total Eelec = "):
            total_elec_energy = float(line.split('=')[1])
        else:
            res_name1, chain1, res_num1, atom_name1, res_name2, chain2, res_num2, atom_name2, dist, elec_energy, vdw_energy = line.split('\t')

            pairs.append((_make_atom(res_name1.strip(), chain1.strip(), int(res_num1), atom_name1.strip(), atom_key),
                          _make_atom(res_name2.strip(), chain2.strip(), int(res_num2), atom_name2.strip(), atom_key),
                          float(dist), float(elec_energy), float(vdw_energy)))

    return pairs, total_vdw_energy, total_elec_energy


def _format_atom(info):
    return " ".join(["{}".format(x) for x in info])


class TestAtomicFeature(unittest.TestCase):
    """Test StructureSimialrity."""

    @staticmethod
    def test_atomic_haddock():

        # in case you change the ref don't forget to:
        # - comment the first line (E0=1)
        # - uncomment the last two lines (Total = ...)
        # - use the corresponding PDB file to test
        REF = 'test/1AK4/atomic_features/ref_1AK4_100w.dat'
        pdb = 'test/1AK4/atomic_features/1AK4_100w.pdb'

        # get the force field included in deeprank
        # if another FF has been used to compute the ref
        # change also this path to the correct one
        FF = pkg_resources.resource_filename(
            'deeprank.features', '') + '/forcefield/'

        # declare the feature calculator instance
        atfeat = AtomicFeature(pdb, select_interface(pdb, 'A', 'B', max_atom_distance=8.5),
                               param_charge=FF + 'protein-allhdg5-4_new.top',
                               param_vdw=FF + 'protein-allhdg5-4_new.param',
                               patch_file=FF + 'patch.top')
        # assign parameters
        atfeat.assign_parameters()

        # only compute the pair interactions here
        total_elec_energy, total_vdw_energy = atfeat.evaluate_pair_interaction()

        # read the ref
        with open(REF, 'rt') as f:
            ref_pairs, ref_total_vdw_energy, ref_total_elec_energy = _parse_ref(f, atfeat.atom_key)

        # Calculate the same sums for the ref as the atomic feature class does:
        ref_elec = {}
        ref_vdw = {}
        for ref_atom1, ref_atom2, dist, elec, vdw in ref_pairs:
            ref_elec[ref_atom1] = ref_elec.get(ref_atom1, 0) + elec
            ref_elec[ref_atom2] = ref_elec.get(ref_atom2, 0) + elec

            ref_vdw[ref_atom1] = ref_vdw.get(ref_atom1, 0) + vdw
            ref_vdw[ref_atom2] = ref_vdw.get(ref_atom2, 0) + vdw

        # Compare the values
        _log.debug("{} ref atoms, {} coulomb atoms, {} vdwaals atoms"
                   .format(len(ref_elec), len(atfeat.feature_data['coulomb']),
                           len(atfeat.feature_data['vdwaals'])))

        eq_(atfeat.feature_data['coulomb'].keys(), ref_elec.keys())
        for ref_atom in ref_elec:
            _close("{} electro energy data".format(_format_atom(ref_atom)), ref_elec[ref_atom], atfeat.feature_data['coulomb'][ref_atom], atol=1E-6)

        eq_(atfeat.feature_data['vdwaals'].keys(), ref_vdw.keys())
        for ref_atom in ref_vdw:
            _close("{} vdwaals energy data".format(_format_atom(ref_atom)), ref_vdw[ref_atom], atfeat.feature_data['vdwaals'][ref_atom], atol=1E-6)

        _close("total electro energy", total_elec_energy, ref_total_elec_energy)
        _close("total vanderwaals energy", total_vdw_energy, ref_total_vdw_energy)

