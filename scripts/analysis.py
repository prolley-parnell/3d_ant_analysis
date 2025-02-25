###
## Functions to analyse the collision class
###

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

    obj_folder = OUTPUT_FOLDER / 'segmentation'
    # obj_folder = OUTPUT_FOLDER / 'segmentation' / SESSION / OBJECT_NAME
    track_folder = OUTPUT_FOLDER / 'tracking'
    # track_folder = OUTPUT_FOLDER / 'tracking' / SESSION /

    track_list = sorted(track_folder.glob(f'{PREFIX}*session{SESSION}*.csv'))

    ## Generate the Animal and Object Instances
    for track in track_folder.iterdir():

        animal = AnimalStruct(args.skeleton, track)