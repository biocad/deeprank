from nose.tools import eq_, ok_
import pdb2sql

from deeprank.selection import get_chains, get_atoms, get_center, InterfaceSelection, MutantSelection


pdb_path = "test/1AK4/atomic_features/1AK4_100w.pdb"


def test_interface():
    selection = InterfaceSelection('A', 'B')

    db = pdb2sql.interface(pdb_path)

    try:
        ok_(len(get_atoms(db, selection, 8.5)) > 0)
        ok_(len(get_chains(db, selection, 8.5)) > 0)
        eq_(len(get_center(db, selection, 8.5)), 3)

    finally:
        db._close()


def test_mutant():
    selection = MutantSelection('A', 10)

    db = pdb2sql.interface(pdb_path)

    try:
        ok_(len(get_atoms(db, selection, 8.5)) > 0)
        ok_(len(get_chains(db, selection, 8.5)) > 0)
        eq_(len(get_center(db, selection, 8.5)), 3)

    finally:
        db._close()
