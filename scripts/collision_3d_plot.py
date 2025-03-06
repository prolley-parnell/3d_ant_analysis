###
## Functions to plot all collisions for a single Animal object, with a single colour for each keypoint or link in collision, plotted on the object
###

###
## Will need to be able to transform the collisions relative to the shape and plot them all on the single object 

import argparse
from pathlib import Path
from src.animal import AnimalStruct
from src.object import CollisionObj
from src.collision import CollisionDetector


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plot antennal contacts over time", exit_on_error=False)
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

    animal_list = []
    ## Generate the Animal and Object Instances
    for i, track in enumerate(track_list):

        animal_list[i] = AnimalStruct(args.skeleton, track)

    collision_detector_list = []
    ## Run the collision check for each frame and store for each animal
    for animal_id, animal in enumerate(animal_list):
        collision_detector_list[animal_id] = CollisionDetector(animal=animal, obj_folder=obj_folder)
