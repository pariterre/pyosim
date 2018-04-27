"""
Example: export forces to sto
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyosim.conf import Conf
from pyosim.obj.analogs import Analogs3dOsim

# path
PROJECT_PATH = Path('../Misc/project_sample')
CALIBRATION_MATRIX = Path('../tests/data/forces_calibration_matrix.csv')

conf = Conf(project_path=PROJECT_PATH)

calibration_matrix = np.genfromtxt(CALIBRATION_MATRIX, delimiter=',')
params = {
    'low_pass_cutoff': 30,
    'order': 4,
    'forces_labels': conf.get_conf_field(participant='dapo', field=['analogs', 'targets'])
}

participants = conf.get_participants_to_process()
# participants = ['dapo']

for iparticipant in participants:
    print(f'\nparticipant: {iparticipant}')
    directories = conf.get_conf_field(participant=iparticipant, field=['analogs', 'data'])
    assigned = conf.get_conf_field(participant=iparticipant, field=['analogs', 'assigned'])

    for itrial in Path(directories[0]).glob('*.c3d'):
        # try participant's channel assignment
        for iassign in assigned:
            # get index where assignment are empty
            nan_idx = [i for i, v in enumerate(iassign) if not v]
            if nan_idx:
                iassign_without_nans = [i for i in iassign if i]
            else:
                iassign_without_nans = iassign

            try:
                forces = Analogs3dOsim.from_c3d(itrial, names=iassign_without_nans, prefix=':')
                if nan_idx:
                    # if there is any empty assignment, fill the dimension with nan
                    for i in nan_idx:
                        forces = np.insert(forces, i, np.nan, axis=1)
                    print(f'\t{itrial.parts[-1]} (NaNs: {nan_idx})')
                else:
                    print(f'\t{itrial.parts[-1]}')

                # check if dimensions are ok
                if not forces.shape[1] == len(iassign):
                    raise ValueError('Wrong dimensions')
                break
            except IndexError:
                forces = []

        # check if there is empty frames
        nan_rows = np.isnan(forces).any(axis=1).ravel()
        print(f'\t\tremoving {nan_rows.sum()} nan frames (indices: {np.argwhere(nan_rows)})')
        forces = forces[..., ~nan_rows]
        # processing (offset during the first second, calibration, )
        forces = forces \
            .center(mu=np.nanmean(forces[..., :int(forces.get_rate)], axis=-1), axis=-1) \
            .squeeze().T.dot(calibration_matrix).T[np.newaxis] \
            .low_pass(freq=forces.get_rate, order=params['order'], cutoff=params['low_pass_cutoff']) \
            .dot(-1)

        norm = Analogs3dOsim(forces.norm().reshape(1, 1, -1))
        idx = norm[0, 0, :].detect_onset(
            threshold=np.nanmean(norm[..., 10:int(forces.get_rate)]) * 10,
            above=int(forces.get_rate) / 2,
            below=3,
            threshold2=np.nanmean(norm[..., :int(forces.get_rate)]) * 4,
            above2=1
        )

        _, ax = plt.subplots(nrows=1, ncols=1)
        norm.plot(ax=ax)
        if idx.shape[0] > 1:
            raise ValueError('more than one onset')
        for (inf, sup) in idx:
            ax.axvline(x=inf, color='g', lw=2, ls='--')
            ax.axvline(x=sup, color='r', lw=2, ls='--')
        plt.show()
        print('')

        # --- Get force onset-offset

    # forces.get_labels = params['forces_labels']
    # sto_filename = PROJECT_PATH / iparticipant / '0_forces' / itrial.parts[-1].replace('c3d', 'sto')
    # forces.to_sto(filename=sto_filename)

import matplotlib.pyplot as plt
import numpy as np

freq = 100
t = np.arange(0, 1, .01)
w = 2 * np.pi * 1
y = np.sin(w * t) + 0.1 * np.sin(10 * w * t)

_, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

ax[0].plot(y, label='courbe 1')
ax[0].plot(y*0.5, label='courbe 2')
ax[0].legend()
ax[0].set_title('Titre 1')

ax[1].plot(y * -2, label='courbe 3')
ax[1].plot(y * 4, label='courbe 4')
ax[1].legend()
ax[1].set_title('Titre 2')

plt.xlabel('Axe X')
plt.ylabel('Axe Y')

plt.legend()
plt.tight_layout()
plt.show()