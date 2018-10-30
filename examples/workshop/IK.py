from pathlib import Path

import opensim as osim

DATA_PATH = Path('/home/romain/Downloads/results')

MODEL = DATA_PATH / 'mars' / '_models' / 'wu_scaled.osim'
MOT = DATA_PATH / 'mars' / '1_inverse_kinematic' / 'wu_MarSF6H2_3.mot'

model = osim.Model(f'{MODEL.resolve()}')
model.initSystem()

motion = osim.Storage(f'{MOT.resolve()}')
first_time = motion.getFirstTime()
last_time = motion.getLastTime()

body_kinematics = osim.BodyKinematics()

body_kinematics.setModel(model)
body_kinematics.setName(body_kinematics.getClassName())
body_kinematics.setOn(True)
body_kinematics.setStepInterval(5)
body_kinematics.setInDegrees(True)
body_kinematics.setStartTime(0)
body_kinematics.setEndTime(2.66)
model.addAnalysis(body_kinematics)

# analysis tool
analyze_tool = osim.AnalyzeTool(model)
analyze_tool.setName(MOT.stem)
analyze_tool.setModel(model)
analyze_tool.setModelFilename(MODEL.stem)

analyze_tool.setInitialTime(first_time)
analyze_tool.setFinalTime(last_time)

analyze_tool.setCoordinatesFileName(f'{MOT.resolve()}')

analyze_tool.setLoadModelAndInput(True)
analyze_tool.setResultsDir('/home/romain/Documents/codes/pyosim/examples/workshop/tests')

analyze_tool.run()
