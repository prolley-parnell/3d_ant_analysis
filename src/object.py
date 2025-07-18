import logging
import concurrent.futures
import trimesh
import os
from pathlib import Path
import toml
import numpy as np

logger = logging.getLogger(__name__)

class CollisionObj:

    def __init__(self, obj_folder: Path | str, units: str = "mm"):
        """
        Initialise an object given by a folder of "obj" files into a dictionary of trimesh objects.
        :param obj_folder: Path to the folder containing the objects
        :param units: Units of the object as the default if none is specified in the file
        """
        if obj_folder is not Path:
            obj_folder = Path(obj_folder)

        self._obj_folder = obj_folder
        self._obj_dict = {}
        self._units = units
        self._colour = trimesh.visual.random_color()
        self._colour[3] = 140 #Change the colour alpha to make the object translucent
        self._read_obj_folder_mt(self._obj_folder)
        self.frames = sorted(self._obj_dict.keys())


    def check_frame_exist(self, frame_idx):
        """ If frame index is not present in the object dictionary, return false"""
        if not self._obj_dict.keys().__contains__(frame_idx):
            logger.debug('Frame index {} not present in the object mesh dictionary \n'.format(frame_idx))
            return False
        return True


    def _read_obj_folder_mt(self, obj_dir_path: Path):
        """ Given a path to a folder containing ".obj" or ".dae" files with the name of the corresponding frame, load these and
        convert to a dict of trimesh"""
        # scan all the file names in this directory
        path_list = obj_dir_path.iterdir()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_path = {executor.submit(single_obj_to_trimesh, obj_path, self._units, self._colour): obj_path for obj_path in path_list}
            for future in concurrent.futures.as_completed(future_to_path):
                path_input = future_to_path[future]
                try:
                    frame_idx, trimesh_obj = future.result()
                except Exception as exc:
                    logger.error('%r generated an exception: %s' % (path_input, exc))
                else:
                    logger.info('Frame index is %d ' % frame_idx)
                    if not trimesh_obj.is_empty:
                        self._obj_dict[frame_idx] = trimesh_obj

    def get_frame_range(self):
        """ Give [min, max] of the frame indices for the stored poses"""
        return [min(self._obj_dict.keys()), max(self._obj_dict.keys())]

    def generate_geometry(self, frame_idx: int) -> trimesh.Trimesh | None:
        """Return a trimesh of the object at the given frame index"""
        if self.check_frame_exist(frame_idx):
            return self._obj_dict[frame_idx]
        else:
            return None

    def generate_scene(self, frame_idx: int):
        """Return a trimesh scene object with the seed in the world frame"""
        scene = trimesh.Scene()
        scene.add_geometry(self.generate_geometry(frame_idx), node_name="object", geom_name=str(frame_idx))
        return scene


class CollisionObjTransform:
    def __init__(self, obj_path: Path, toml_path: Path, units: str = "m"):
        """
        Initialise an object given by a folder of "obj" files into a dictionary of trimesh objects.
        Convert from dict of trimesh into a single mesh and a dict of transforms
        :param obj_path: Path to the object file
        :param toml_path: Path to the toml file with object transformations
        """
        if obj_path is not Path:
            obj_path = Path(obj_path).resolve()

        self._units = units
        self._colour = trimesh.visual.random_color()
        self._colour[3] = 140 #Change the colour alpha to make the object translucent

        self._obj_path = obj_path
        self._reference_frame, self._obj_mesh = single_obj_to_trimesh(self._obj_path, self._units, self._colour)


        if toml_path is not Path:
            toml_path = Path(toml_path).resolve()
        self._obj_transforms = self.load_transform_toml(toml_path)

    @staticmethod
    def load_transform_toml(toml_path: Path):
        ## Assumes the saved file format surrounds each number in the array with 'np.float64(0.00)'
        def toml_tf_format(_dict=dict):
            dict_out = dict()
            for k, v in _dict.items():
                new_key = int(k.split('_')[-1])
                dict_out[new_key] = np.array(v, dtype=np.float64)

            return dict_out

        with open(toml_path, 'r') as f:
            toml_str = f.read()

        transform_dict = toml_tf_format(toml.loads(toml_str.replace('np.float64(', '').replace(')', '')))
        return transform_dict

    def obj_mesh(self):
        return self._obj_mesh

    def check_frame_exist(self, frame_idx):
        """ If frame index is not present in the object dictionary, return false"""
        if not self._obj_transforms.keys().__contains__(frame_idx):
            logger.debug('Frame index {} not present in the object mesh dictionary \n'.format(frame_idx))
            return False
        return True

    def get_frame_range(self):
        """ Give [min, max] of the frame indices for the stored poses"""
        return [min(self._obj_transforms.keys()), max(self._obj_transforms.keys())]

    def generate_transform(self, frame_idx: int):
        """Return the transform of the object at the given frame index"""
        if self.check_frame_exist(frame_idx):
            return self._obj_transforms[frame_idx]
        else:
            return None

    def generate_scene(self, frame_idx: int):
        """Return a trimesh scene object with the seed in the world frame"""
        scene = trimesh.Scene()
        scene.add_geometry(self._obj_mesh, node_name="object", geom_name=str(frame_idx), transform=self.generate_transform(frame_idx))
        return scene


def single_obj_to_trimesh(obj_path: Path, _units: str = "m", _colour: [int] = None):
    """ Convert the dae or obj at the given path to a trimesh object and return the object and frame index """
    if os.path.isdir(obj_path):
        raise Exception("Folder {} ignored".format(obj_path))
    if obj_path.suffix in [".dae", ".DAE"]:
        frame_index = int(os.path.splitext(obj_path)[-2].split('frame')[-1]) # Add to only get the number next to the dae extension (e.g. '240905-1616_seed_session28_frame568.dae')
        trimesh_obj = trimesh.load_mesh(obj_path, 'dae', process=False)
    elif obj_path.suffix in [".obj", ".OBJ"]:
        frame_index = int(os.path.splitext(obj_path)[-2].split('_')[-1]) # Add to only get the number next to the obj extension (e.g. 'seed_200.obj')
        trimesh_obj = trimesh.load_mesh(obj_path, 'obj', process=False)
    else:
        raise Exception("File {} ignored, not .dae or .obj file".format(obj_path))

    if trimesh_obj.is_empty:
        raise Exception("Mesh {} ignored, file empty".format(obj_path))
    if trimesh_obj.units is None:
        trimesh_obj.units = _units
    if _colour is not None:
        trimesh_obj.visual.face_colors = np.array([_colour for i in trimesh_obj.faces])
    return frame_index, trimesh_obj