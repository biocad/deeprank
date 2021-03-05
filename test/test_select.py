from nose.tools import eq_, ok_

from deeprank.selection import select_interface, select_residue_environment


pdb_path = "test/1AK4/atomic_features/1AK4_100w.pdb"


def test_interface():
    selection = select_interface(pdb_path, 'A', 'B')

    ok_(len(selection.atoms) > 0)
    eq_(len(selection.center_position), 3)


def test_residue():
    selection = select_residue_environment(pdb_path, 'A', 10)

    ok_(len(selection.atoms) > 0)
    eq_(len(selection.center_position), 3)
