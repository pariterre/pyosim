import numpy as np

import opensim as osim

subject = "mars"
pyosim_example_path = "/home/romain/Downloads"
osim_model = "wu_scaled.osim"
Qfile = f"{pyosim_example_path}/results/{subject}/1_inverse_kinematic/wu_MarSF6H2_3.mot"
muscle_excitation = np.ones([29]) * 0

# Load the model
model_path = f"{pyosim_example_path}/results/{subject}/_models/{osim_model}"

# Prepare a model
model = osim.Model(model_path)
state = model.initSystem()

# Read the data
data_storage = osim.Storage(Qfile)

nrows = data_storage.getSize()
x_dot = np.zeros((nrows, 96))

for irow in range(nrows):
    # Store Q in state
    state.setQ(data_storage.getStateVector(irow).getData().getAsVector())

    # update the state velocity calculations
    model.computeStateVariableDerivatives(state)

    num_var = state.getNY()
    for icol in range(num_var):
        x_dot[irow, icol] = state.getYDot().get(icol)
