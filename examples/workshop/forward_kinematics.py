import biorbd
import numpy as np
from scipy.io import loadmat

from pyosim import Markers3dOsim

trial = 'MarSF12H6_1'

MODEL = biorbd.s2mMusculoSkeletalModel(f'/media/romain/F/Data/Shoulder/Lib/IRSST_MarSd/Model_2/Model.s2mMod')
Q2_FILE = f'/media/romain/E/Projet_Reconstructions/DATA/Romain/IRSST_MarSd/trials/{trial}_MOD2.1_rightHanded_GenderF_IRSST_MarSd_.Q2'
Q2_DATA = loadmat(Q2_FILE)

tags_name = [
    'ASISl', 'ASISr', 'PSISl', 'PSISr',
    'STER', 'STERl', 'STERr', 'T1', 'T10', 'XIPH',
    'CLAVm', 'CLAVl', 'CLAV_ant', 'CLAV_post', 'CLAV_SC',
    'ACRO_tip', 'SCAP_AA', 'SCAPl', 'SCAPm', 'SCAP_CP', 'SCAP_RS', 'SCAP_SA', 'SCAP_IA', 'CLAV_AC',
    'DELT', 'ARMl', 'ARMm', 'ARMp_up', 'ARMp_do', 'EPICl', 'EPICm',
    'LARMm', 'LARMl', 'LARM_elb', 'LARM_ant',
    'STYLr', 'STYLr_up', 'STYLu', 'WRIST',
    'INDEX', 'LASTC', 'MEDH', 'LATH'
]

TRC_FILE = f'/home/romain/Downloads/results/mars/0_markers/{trial}.trc'
x = Markers3dOsim.from_trc(TRC_FILE)
labels = np.array(x.get_labels)
x.get_rate = 1 / (x.get_time_frames[1] - x.get_time_frames[0])
x.get_unit = 'mm'


x.fill_values().low_pass(100, 2, 10).to_trc(f'/home/romain/Downloads/results/mars/0_markers/comparison/{trial}_interp_filter.trc')

new_markers = {}
for iframe in range(Q2_DATA['Q2'].shape[1]):
    Q = biorbd.s2mGenCoord(Q2_DATA['Q2'][:, iframe])
    T = MODEL.Tags(MODEL, Q)

    for t, itags_name in zip(T, tags_name):
        if itags_name not in new_markers:
            new_markers[itags_name] = t.get_array()
        else:
            new_markers[itags_name] = np.c_[new_markers[itags_name], t.get_array()]

for key, value in new_markers.items():
    x[:-1, labels == key, :] = value.reshape(3, 1, -1) * 1000  # we multiply to convert m to mm

x[:, 43:, :] = x[:, 43:, :].fill_values()
x.low_pass(100, 2, 10).to_trc(f'/home/romain/Downloads/results/mars/0_markers/comparison/{trial}_kalman_filter.trc')



