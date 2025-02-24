import logging
import concurrent.futures
import trimesh
import os
import numpy as np

# import trimesh.viewer

logger = logging.getLogger(__name__)


class CollisionObj:

    def __init__(self, obj_folder: str, units: str = "mm"):
        '''
        Initialise an object given by a folder of "obj" files into a dictionary of trimesh objects.
        :param obj_folder: Path to the folder containing the objects
        '''
        self._obj_folder = obj_folder
        self._obj_dict = {}
        self._units = units
        self._read_obj_folder_mt(self._obj_folder)

    def check_frame_exist(self, frame_idx):
        ''' If frame index is not present in the object dictionary, return false'''
        if not self._obj_dict.keys().__contains__(frame_idx):
            logger.debug('Frame index {} not present in the object mesh dictionary \n'.format(frame_idx))
            return False
        return True

    def _read_obj_folder(self, obj_dir_path: str):
        ''' Given a path to a folder containing ".obj" files with the name of the corresponding frame, load these and
        convert to a dict of trimesh'''
        # scan all the OBJ file names in this directory
        path_list = os.listdir(obj_dir_path)

        for p in path_list:
            frame_idx, trimesh_obj = self._single_obj_to_trimesh(p)
            self._obj_dict[frame_idx] = trimesh_obj


    def _read_obj_folder_mt(self, obj_dir_path: str):
        ''' Given a path to a folder containing ".obj" or ".dae" files with the name of the corresponding frame, load these and
        convert to a dict of trimesh'''
        # scan all the OBJ file names in this directory
        path_list = os.listdir(obj_dir_path)

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


    def _single_obj_to_trimesh(self, obj_path: str):
        ''' Convert the dae at the given path to a trimesh object and return the object and frame index '''
        if os.path.isdir(self._obj_folder + obj_path):
            raise Exception("Folder {} ignored".format(self._obj_folder + obj_path))
        if os.path.splitext(obj_path)[-1] in [".dae", ".DAE"] and not obj_path.startswith('.'):
            frame_index = int(os.path.splitext(obj_path)[-2].split('_')[-1]) # Add to only get the number next to the obj extension
            trimesh_obj = trimesh.load_mesh(self._obj_folder + obj_path, 'dae')
        elif os.path.splitext(obj_path)[-1] in [".obj", ".OBJ"] and not obj_path.startswith('.'):
            frame_index = int(os.path.splitext(obj_path)[-2].split('_')[-1]) # Add to only get the number next to the obj extension
            trimesh_obj = trimesh.load_mesh(self._obj_folder + obj_path, 'obj')
        else:
            raise Exception("File {} ignored, not .dae or .obj file".format(self._obj_folder + obj_path))

            # trimesh_obj.vertices = np.column_stack((trimesh_obj.vertices[:, 1], trimesh_obj.vertices[:, 0], trimesh_obj.vertices[:, 2]))
        if trimesh_obj.is_empty:
            raise Exception("Mesh {} ignored, file empty".format(self._obj_folder + obj_path))
        if trimesh_obj.units is None:
            trimesh_obj.units = self._units
        return frame_index, trimesh_obj

    def get_frame_range(self):
        ''' Give [min, max] of the frame indices for the stored poses'''
        return [min(self._obj_dict.keys()), max(self._obj_dict.keys())]

    def generate_geometry(self, frame_idx: int):
        '''Return a trimesh of the object at the given frame index'''
        if self.check_frame_exist(frame_idx):
            return self._obj_dict[frame_idx]
        else:
            return None

    def generate_scene(self, frame_idx: int):
        '''Return a trimesh scene object with the seed in the world frame'''
        scene = trimesh.Scene()
        scene.add_geometry(self.generate_geometry(frame_idx), node_name="object", geom_name=str(frame_idx))
        return scene
