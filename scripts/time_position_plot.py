### Script to plot the coordinates of one, or multiple keypoints of an Animal object across the frames with X, Y and Z as individual Y axes and frame number as the X axis.

import argparse
from pathlib import Path

from src.animal import AnimalList
from scripts.tools.x_y_z_plot import xyzPlot

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plot keypoint positions over time", exit_on_error=False)
    parser.add_argument("folder", help="Path to the datetime folder")
    parser.add_argument("prefix", type=str, help="Datetime prefix (example: '240905-1616')")
    parser.add_argument("session", type=int, help="Session number (example: 28)")
    parser.add_argument("skeleton", type=str, help="skeleton.toml (An example in the top level of this repo)")


    try:
        args = parser.parse_args()
    except:
        # We're running from the console
        from dataclasses import dataclass
        @dataclass
        class args:
            folder = Path('/Users/persie/PhD_Code/3d_ant_data_rle')
            prefix = '240905-1616'
            session = 28
            skeleton = '../skeleton.toml'

    PREFIX = args.prefix
    FOLDER = Path(args.folder) / PREFIX
    SESSION = args.session

    OUTPUT_FOLDER = FOLDER / 'outputs'

    obj_folder = OUTPUT_FOLDER / 'segmentation' / str(SESSION)
    track_folder = OUTPUT_FOLDER / 'tracking' / str(SESSION)

    track_list = sorted(track_folder.glob(f'{PREFIX}*session{SESSION}*.csv'))


    ## Generate the Animal and Object Instances

    animals_list = AnimalList(args.skeleton, track_folder, session_number=SESSION)

    collision_detector_list = []
    ## Run the collision check for each frame and store for each animal
    for animal_id, animal in enumerate(animal_list):
        collision_detector_list[animal_id] = CollisionDetector(animal=animal, obj_folder=obj_folder)
