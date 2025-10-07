#Plot as a scatter the duration of time spend exploring the object in frames in relation to the size of the head of the ant
from scripts.tools.ground_truth_toml_reader import GroundTruth
from src.animal import AnimalList
from pathlib import Path
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


def scale_from_position(input_position: DataFrame):
    #For all columns that are not Nan at all, find height of triangle a, b, c and find the median of them all, head length and eye distance

    valid_input = input_position.dropna(axis=1)

    inter_eye_dist = (valid_input.loc['eye_R'] - valid_input.loc['eye_L']).astype(np.float64)

    magnitude_inter_eye = np.sqrt(np.square(inter_eye_dist).sum(axis=0))

    return magnitude_inter_eye.median()

data_folder = Path("/Users/persie/Desktop/Dataset_for_upload/Data/")

session_list = [6,10,11,13,17,19,20,21,22,23,26,28,29,30]
prefix = "240905-1616"
skeleton_toml_path = "../skeleton.toml"

node_list = ['eye_R', 'eye_L', 'neck']

data_out = {}

#For each ant in the sample
for session in session_list:

    gt = GroundTruth(path=data_folder, prefix=prefix, session_number=session)

    animal_pkl = data_folder / prefix / "outputs" / "tracking" / f"linked_tracks_session{session}.pkl"

    # Get head size
    animal_list = AnimalList(toml_path=skeleton_toml_path, pose_pkl=animal_pkl, track_number=gt["reliable_track"])

    #Get animal
    a = animal_list.animal(gt["reliable_track"][0])

    #Get size of animal
    position_df, kp_in_df = a.get_xyz_df(node_list=node_list)
    median_inter_eye = scale_from_position(position_df)

    print(f'Median Inter Eye Distance for session {session}: {median_inter_eye}')

    #Get interaction duration
    interaction_duration = np.float64(gt["grasp"] - gt["touch"])*0.01

    data_out[session] = [median_inter_eye, interaction_duration]

data_out_df = DataFrame(data_out, index=['x', 'y'])

plt.scatter(data_out_df.loc['x'], data_out_df.loc['y'])
plt.xlabel("Median Inter Eye Distance in millimeters")
plt.ylabel("Duration in seconds")
plt.xticks(ticks=np.arange(0.9, 2, 0.1))
plt.yticks(ticks=np.arange(0, 35, 5))
plt.title("Ant Size against time to grasp")
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
plt.grid(visible=True, which='minor', linestyle='-', linewidth=0.1)
# plt.savefig("/Users/persie/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD/Figures/Thesis Figures/Experiment Analysis/size_to_duration.svg")
plt.show()