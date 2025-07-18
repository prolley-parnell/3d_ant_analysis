import logging
from typing import Optional
from pathlib import Path

from src.animal import AnimalStruct, AnimalList
from src.object import CollisionObj, CollisionObjTransform


logger = logging.getLogger(__name__)

class InstanceLoader:

    def __init__(self,
                 animal: Optional[AnimalStruct | AnimalList]  = None ,
                 obj: Optional[CollisionObj | CollisionObjTransform]  = None,
                 obj_folder: Optional[str | Path] = None,
                 obj_ref_frame: Optional[int] = None,
                 obj_transform_toml: Optional[str | Path] = None,
                 skeleton_toml_path: Optional[str | Path] = None,
                 pose_csv: Optional[str | Path] = None,
                 track_folder: Optional[str | Path] = None,
                 session_number: Optional[int] = None,
                 track_number: Optional[list[int]] = None,
                 prefix: Optional[str] = None,
                 ):

        self._animal_list = None
        if animal is None:
            if skeleton_toml_path is None:
                raise Exception("No skeleton toml included")
            else:
                if pose_csv is not None and track_folder is not None:
                    raise Exception("Only the track folder or the individual track pose csv can be included.")
                elif pose_csv is not None:
                    self._animal = AnimalStruct(skeleton_toml_path, pose_csv)
                    self._animal_list = AnimalList([self._animal])

                elif track_folder is not None:
                    if session_number is None:
                        raise Exception("No session number included")
                    else:
                        self._animal_list = AnimalList(
                            toml_path=skeleton_toml_path,
                            csv_folder=track_folder,
                            session_number=session_number,
                            track_number=track_number)
                else:
                    raise Exception("No animal, single track pose csv, or track folder included")

        elif type(animal) is AnimalList:
            self._animal_list = animal

        elif type(animal) is AnimalStruct:
            self._animal = animal
            self._animal_list = AnimalList([self._animal])

        if obj is None:
            if obj_folder is not None:
                if obj_folder is not Path:
                    obj_folder = Path(obj_folder).resolve()

                if obj_ref_frame is not None and session_number is not None:
                    obj_path = sorted(obj_folder.glob(f"*{session_number}_frame{str(obj_ref_frame)}.dae"))
                    if len(obj_path) == 0:
                        logger.error(f"No paths matching {obj_folder} were found")
                        return
                    elif len(obj_path) > 1:
                        logger.error(f"Multiple paths matching {obj_folder} were found")
                        return
                    else:
                        obj_path = obj_path[0]

                    if obj_transform_toml is None:
                        if prefix is not None:
                            obj_transform_toml = Path(obj_folder.parent) / f"segmentation/{prefix}_seed_session{session_number}.toml"
                        else:
                            logger.error(f"No transformation toml included")


                    self._obj = CollisionObjTransform(obj_path, obj_transform_toml)
                else:
                    self._obj = CollisionObj(obj_folder)
            else:
                raise Exception("No object included, and no path included")
        else:
            self._obj = obj

    @property
    def animal_list(self) -> AnimalList:
        return self._animal_list

    @property
    def obj_list(self) -> list[CollisionObj]:
        return [self._obj]