import sys
import _0_project
# import _1_markers
# import _2_emg
# import _3_forces
# import _4_scaling
# import _5_inverse_kinematics
# import _6_inverse_dynamics
# import _7_static_optimization
# import _8_muscle_analysis
# import _9_joint_reaction
#


def main(participant_to_do):
    _0_project.main(specific_participant=participant_to_do, erase_previous_project=False)


if __name__ == "__main__":
    main(sys.argv[1])

