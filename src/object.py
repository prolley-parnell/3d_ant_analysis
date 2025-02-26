import logging
import concurrent.futures
import trimesh
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CollisionObj:

    def __init__(self, obj_folder: Path | str, units: str = "m"):
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
            future_to_path = {executor.submit(self._single_obj_to_trimesh, obj_path): obj_path for obj_path in path_list}
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


    def _single_obj_to_trimesh(self, obj_path: Path):
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
            trimesh_obj.units = self._units
        return frame_index, trimesh_obj

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
    def __init__(self, obj_folder: Path, units: str = "m"):
        """
        Initialise an object given by a folder of "obj" files into a dictionary of trimesh objects.
        Convert from dict of trimesh into a single mesh and a dict of transforms
        :param obj_folder: Path to the folder containing the objects
        """
        self._obj_folder = obj_folder
        self._units = units
        co = CollisionObj(obj_folder, units)
        self.obj_mesh, self._obj_transforms = convert_collision_obj(co)



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
        scene.add_geometry(self.obj_mesh, node_name="object", geom_name=str(frame_idx), transform=self.generate_transform(frame_idx))
        return scene


def convert_collision_obj(co: CollisionObj):
    """ Convert from a CollisionObject with a dict of objects, into a single trimesh object and a dict of transforms"""

    output_transforms = dict()

    sorted_frames = co.frames
    origin = trimesh.creation.axis(origin_size=1, axis_length=4)
    geom_prior = co.generate_geometry(sorted_frames[0])
    output_transforms[sorted_frames[0]] = geom_prior.principal_inertia_transform

    origin.apply_transform(output_transforms[sorted_frames[0]])
    
    for frame in sorted_frames[1:14]:

        geom_base = co.generate_geometry(frame)
        print(geom_prior.principal_inertia_vectors)
        T, cost = geom_prior.register(geom_base)
        geom_prior.apply_transform(T)
        origin.apply_transform(T)
        print(cost)
        output_transforms[frame] = origin.principal_inertia_transform

    output_mesh = geom_prior

    return output_mesh, output_transforms