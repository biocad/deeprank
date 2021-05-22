import pdb2sql
from pdb2sql import interface, transform
from pdb2sql.superpose import get_trans_vect, get_rotation_matrix, superpose_selection
import itertools



class MultiChainStructureSimilarity(pdb2sql.StructureSimilarity):
    def __init__(self, decoy, ref, chains1, chains2, verbose=False):
        super().__init__(decoy, ref, verbose)
        self.chains1 = chains1
        self.chains2 = chains2

    def compute_irmsd_pdb2sql_multi(
            self,
            cutoff=10,
            method='svd',
            izone=None,
            exportpath=None):
        sql_decoy = interface(self.decoy)
        sql_ref = interface(self.ref)


        chains_decoy = sql_decoy.get_chains()
        chains_ref = sql_ref.get_chains()

        if chains_decoy != chains_ref:
            raise ValueError(
                'Chains are different in decoy and reference structure')

        if izone is None:

            contact_ref = dict()
            for ch in chains_ref:
                contact_ref[ch] = []

            for ch1, ch2 in itertools.product(self.chains1, self.chains2):
                contact_ref_temp = sql_ref.get_contact_atoms(
                    cutoff=cutoff,
                    extend_to_residue=True,
                    chain1=ch1,
                    chain2=ch2)
                contact_ref[ch1] = list(set(contact_ref[ch1] + contact_ref_temp[ch1]))
                contact_ref[ch2] = list(set(contact_ref[ch2] + contact_ref_temp[ch2]))

            index_contact_ref = []
            for v in contact_ref.values():
                index_contact_ref += v

            xyz_contact_ref = sql_ref.get(
                'x,y,z', rowID=index_contact_ref)
            data_contact_ref = sql_ref.get(
                'chainID,resSeq,resName,iCode,name',
                rowID=index_contact_ref)
            xyz_decoy = sql_decoy.get('x,y,z')
            data_decoy = sql_decoy.get('chainID,resSeq,resName,iCode,name')

            xyz_contact_decoy = []
            index_contact_decoy = []
            clean_ref = False
            for iat, atom in enumerate(data_contact_ref):

                try:
                    index = data_decoy.index(atom)
                    index_contact_decoy.append(index)
                    xyz_contact_decoy.append(xyz_decoy[index])
                except Exception:
                    xyz_contact_ref[iat] = None
                    index_contact_ref[iat] = None
                    clean_ref = True

            if clean_ref:
                xyz_contact_ref = [
                    xyz for xyz in xyz_contact_ref if xyz is not None]
                index_contact_ref = [
                    ind for ind in index_contact_ref if ind is not None]

            chain_decoy = list(
                set(sql_decoy.get('chainID', rowID=index_contact_decoy)))
            chain_ref = list(
                set(sql_ref.get('chainID', rowID=index_contact_ref)))

            if len(chain_decoy) < 1 or len(chain_ref) < 1:
                raise ValueError(
                    'Error in i-rmsd: only one chain represented in one chain')

            # get the translation so that both A chains are centered
            tr_decoy = get_trans_vect(xyz_contact_decoy)
            tr_ref = get_trans_vect(xyz_contact_ref)

            # translate everything
            xyz_contact_decoy += tr_decoy
            xyz_contact_ref += tr_ref

            # get the ideql rotation matrix
            # to superimpose the A chains
            rot_mat = get_rotation_matrix(
                xyz_contact_decoy,
                xyz_contact_ref,
                method=method)

            # get the ideql rotation matrix
            # to superimpose the A chains
            rot_mat = get_rotation_matrix(
                xyz_contact_decoy,
                xyz_contact_ref,
                method=method)

            # rotate the entire fragment
            xyz_contact_decoy = transform.rotate(
                xyz_contact_decoy, rot_mat, center=self.origin)

            irmsd = self.get_rmsd(xyz_contact_decoy, xyz_contact_ref)

            if exportpath is not None:
                # update the sql database
                sql_decoy.update_xyz(
                    xyz_contact_decoy, rowID=index_contact_decoy)
                sql_ref.update_xyz(
                    xyz_contact_ref, rowID=index_contact_ref)

                sql_decoy.exportpdb(
                    exportpath + '/irmsd_decoy.pdb',
                    rowID=index_contact_decoy)
                sql_ref.exportpdb(
                    exportpath + '/irmsd_ref.pdb',
                    rowID=index_contact_ref)

            # close the db
            sql_decoy._close()
            sql_ref._close()

            return irmsd

