import numpy as np
import warnings


def __compute_target__(decoy, targrp, chains1, chains2):
    tarname = 'BIN_CLASS'

    molgrp = targrp.parent
    molname = molgrp.name

    if tarname in targrp.keys():
        del targrp[tarname]
        warnings.warn(f"Removed old {tarname} from {molname}")

    targrp.create_dataset('BIN_CLASS', data=np.array(1))

def __compute_target_ram__(decoy, targrp, chains1, chains2):
    tarname = 'BIN_CLASS'

    targrp[tarname] = np.array(1)
