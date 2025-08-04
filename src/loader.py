import logging
from typing import Optional
from pathlib import Path

from src.animal import AnimalStruct, AnimalList
from src.object import CollisionObj, CollisionObjTransform


logger = logging.getLogger(__name__)

class InstanceLoader:

    def __init__(self,
                 animal: Optional[AnimalStruct | AnimalList]  = None ,
                 data_folder: Optional[str | Path] = None,
                 obj: Optional[CollisionObj | CollisionObjTransform]  = None,
                 obj_ref_frame: Optional[int] = None,
                 obj_transform_toml: Optional[str | Path] = None,
                 skeleton_toml_path: Optional[str | Path] = None,
                 animal_pkl: Optional[str | Path] = None,
                 session_number: Optional[int] = None,
                 track_number: Optional[list[int] | list[str]] = None,
                 prefix: Optional[str] = None,
                 ):

        self._animal_list = None
        if data_folder is not Path:
            data_folder = Path(data_folder).resolve()

        if animal is None:
            if skeleton_toml_path is None:
                raise Exception("No skeleton toml included")
            else:
                if animal_pkl is None:
                    if data_folder is not None and session_number is not None and prefix is not None:
                        animal_pkl = data_folder / prefix / "outputs" / "tracking" / f"linked_tracks_session{session_number}.pkl"
                    else:
                        raise Exception("No animal pkl included")
                self._animal_list = AnimalList(toml_path=skeleton_toml_path, pose_pkl=animal_pkl,
                                               track_number=track_number)
                self._animal = self._animal_list.animal

        elif type(animal) is AnimalList:
            self._animal_list = animal

        elif type(animal) is AnimalStruct:
            self._animal = animal
            self._animal_list = AnimalList([self._animal])

        if obj is None:
            if data_folder is not None and prefix is not None:
                obj_folder = data_folder / prefix / "outputs" / "segmentation"

                if obj_folder is not Path:
                    obj_folder = Path(obj_folder).resolve()

                if obj_ref_frame is not None and session_number is not None:
                    obj_path = sorted(obj_folder.glob(f"*{session_number}_frame{str(obj_ref_frame)}.dae"))
                    if len(obj_path) == 0:
                        logger.error(f"No dae with Session {session_number} and Ref Frame {obj_ref_frame} were found")
                        return
                    elif len(obj_path) > 1:
                        logger.error(f"Multiple dae with Session {session_number} and Ref Frame {obj_ref_frame} were found")
                        return
                    else:
                        obj_path = obj_path[0]

                    if obj_transform_toml is None:
                        if prefix is not None:
                            obj_transform_toml = Path(obj_folder) / f"{prefix}_seed_session{session_number}.toml"
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