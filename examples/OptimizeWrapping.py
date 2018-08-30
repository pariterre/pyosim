import opensim as osim
import numpy as np

from pyosim import MuscleAnalysis
from pyomeca import Analogs3d
from project_conf import MODELS_PATH, PROJECT_PATH, TEMPLATES_PATH

# Optimization strategy
#   optimize
#       the wrapping objects dimensions;
#   in order to
#       maximize R² with the template on muscle lengths and muscular moment arms;
#       minimize the difference with the first guesses;
#   such as
#       there is no discontinuity in muscle lengths;
#
# OBJECTIVE FUNCTION
#   Load the mot file (motion that recreates normal upper limb activity)
#   Perform the muscle analysis
#   Load the results
#   Compute the R² of moment arms and muscle lengths between the scaled model and the non-scaled model
#   Minimize the R² and the difference with the first guess
#
# CONSTRAINT FUNCTION
#   Load the muscle analysis results
#   Find any discontinuities (velocity of muscle lengths > mean_rms * 10)


# NOTE : For some reason, in debug opensim writes its sto file comma decimal instead of point decimal...
model = 'wu'
subject = 'mars'

template_model = f'{MODELS_PATH}/{model}.osim'
subject_model = f'{PROJECT_PATH}/{subject}/_models/{model}_scaled.osim'
mot_file = f'{model}_optimMotion.mot'


def generate_motion(dofs, resolution=100):
    # First time was in the hope to recursively call  the function do an exhaustive scan of the ranges of motion.
    # Unless better idea comes, it raises an out of memory because the matrix is [resolution^len(dofs)] long

    # if first_time:
    final_array = np.ndarray((resolution, len(dofs)+1))

    final_array[:, 14] = np.linspace(0, np.pi/2, resolution)

    # if first_time:
    final_array[:, 0] = np.linspace(0, (resolution-1)/100, resolution)
    return final_array


def generate_mot_file(path_to_save, data, dofs):
    if len(dofs) != data.shape[1] - 1:  # -1 for the time column
        raise ValueError("Wrong number of dofs")

    # Prepare the header
    header = list()
    header.append(f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns={data.shape[1]}\ninDegrees=no\nendheader\n")
    header.append("time\t")
    for dof in dofs:
        header.append(f"{dof}\t")
    header = ''.join(header)

    # Write the data
    np.savetxt(path_to_save, data, delimiter='\t', header=header, comments='')


def perform_muscle_analysis(path_model, output_path='temp_optim_wrap'):
    global analyse_must_be_perform, data_of_the_subject
    if analyse_must_be_perform:
        path_kwargs = {
            'model_input': path_model,
            'xml_input': f"{(TEMPLATES_PATH / model).resolve()}_ma_optimWrap.xml",
            'xml_output': f"{(PROJECT_PATH / subject / '_xml' / model).resolve()}_ma_optimWrap.xml",
            'sto_output': f"{(PROJECT_PATH / subject / output_path).resolve()}",
            'enforce_analysis': True
        }

        MuscleAnalysis(
            **path_kwargs,
            mot_files=f'{PROJECT_PATH}/{subject}/temp_optim_wrap/{mot_file}',
            prefix=model,
            low_pass=-1,
            remove_empty_files=True,
            multi=False
        )
        analyse_must_be_perform = False
        data_of_the_subject = Analogs3d.from_csv(
            f"{(PROJECT_PATH / subject / output_path / f'{mot_file[:-4]}_MuscleAnalysis_Length.sto').resolve()}",
            delimiter='\t', time_column=0, header=9,
            first_column=1, first_row=10).derivative().abs()[:, :, 1:-1]


def continuous_muscle_constraint():
    # Inequality constraint.
    # If muscle length top 2% change in velocity is lower than mean_rms * 10, we assume all muscles are continuous
    perform_muscle_analysis(subject_model)

    mean_rms = np.mean(data_of_the_subject.rms())
    return np.percentile(-1*data_of_the_subject, 2, axis=2) + mean_rms * 10


def objective_function():
    # MODIFY SUBJECT_MODEL : TODO
    # osim_model.get_BodySet().get(0).get_WrapObjectSet().get(5).set

    # Compute R²
    perform_muscle_analysis(subject_model)
    corrcoef = list()
    for dof in range(data_of_the_subject.shape[1]):
        coef = np.corrcoef(data_of_the_subject[0, dof, :], data_template[0, dof, :])[1, 0]
        if np.isnan(coef):
            val = 0
        else:
            val = 1 - coef * coef  # R^2
        corrcoef.append(val)
    return corrcoef


# Load osim files
osim_model = osim.Model(subject_model)

# Generate a mot file which can create muscles discontinuity and get all degrees of freedom of the model
dofs = list()
for i in range(osim_model.get_JointSet().getSize()):
    for j in range(osim_model.get_JointSet().get(i).numCoordinates()):
        dofs.append(osim_model.get_JointSet().get(i).get_coordinates(j).getName())
generate_mot_file(f'{PROJECT_PATH}/{subject}/temp_optim_wrap/{mot_file}', generate_motion(dofs), dofs)

# Do a first muscle analysis
data_of_the_subject = Analogs3d(np.ndarray((0, 0)))
analyse_must_be_perform = True
perform_muscle_analysis(template_model, 'template_temp_optim_wrap')
data_template = data_of_the_subject
analyse_must_be_perform = True
perform_muscle_analysis(subject_model)

# Test constraint function
print(objective_function())
print(continuous_muscle_constraint())



## ERREUR DE SEGMENT DÛ À MATPLOTLIB... VÉRIFIER POURQUOI..



import pygmo as pg

# 1 - Instantiate a pygmo problem constructing it from a UDP
# (user defined problem).
prob = pg.problem(pg.schwefel(30))

# 2 - Instantiate a pagmo algorithm
algo = pg.algorithm(pg.sade(gen=100))

# 3 - Instantiate an archipelago with 16 islands having each 20 individuals
archi = pg.archipelago(16, algo=algo, prob=prob, pop_size=20)

# 4 - Run the evolution in parallel on the 16 separate islands 10 times.
archi.evolve(10)

# 5 - Wait for the evolutions to be finished
archi.wait()

# 6 - Print the fitness of the best solution in each island
res = [isl.get_population().champion_f for isl in archi]

print(res)