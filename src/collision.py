import logging
from typing import Optional

import numpy as np
import trimesh

from src.animal import AnimalStruct
from src.object import CollisionObj


logger = logging.getLogger(__name__)


class CollisionDetector:

    def __init__(self,
                 animal: Optional[AnimalStruct]  = None ,
                 obj: Optional[CollisionObj]  = None,
                 obj_folder: Optional[str] = None,
                 skeleton_toml_path: Optional[str] = None,
                 pose_csv: Optional[str] = None
                 ):

        if animal is None:
            if skeleton_toml_path is not None and pose_csv is not None:
                self.animal = AnimalStruct(skeleton_toml_path, pose_csv)
            else:
                raise Exception("No animal, skeleton_toml_path, or pose_csv included")
        else:
            self.animal = animal

        if obj is None:
            if obj_folder is not None:
                self.obj = CollisionObj(obj_folder)
            else:
                raise Exception("No object included, and no path included")
        else:
            self.obj = obj


    def visualise_collision(self, frame_idx: int):
        ''' Give a frame index, calculate collision points then visualise'''

        location_array, normal_array = self.get_collisions(frame_idx)
        if len(location_array) == 0:
            logger.info("No collisions found for %d" % frame_idx)
            ray_visualise = []
        else:
            ray_visualise = trimesh.load_path(
                np.hstack((location_array, location_array + normal_array * 5.0)).reshape(-1, 2, 3)
            )
        ax = trimesh.creation.axis(10)
        scene = trimesh.Scene([self.obj.generate_geometry(frame_idx), ax, ray_visualise])

        return scene


    def get_collisions(self, frame_idx: int):
        '''
        Given a frame index, calculate and return the collisions with the object and the surface normal at the
        point of collision.
        '''
        if not self.check_frame_exist(frame_idx):
            logger.warning('Could not find collision for frame index {} because both object and animal are not in this frame'.format(frame_idx))
            return None

        pose_ray_dict = self.animal.get_pose_ray(frame_idx)
        #ri = trimesh.ray.ray_pyembree.RayMeshIntersector(self._obj_dict[frame_idx])
        ri = trimesh.ray.ray_triangle.RayMeshIntersector(self.obj.generate_geometry(frame_idx))

        index_tri, index_ray, location = ri.intersects_id(
            ray_origins=[pose_ray_dict[link]['origin'] for link in pose_ray_dict.keys()],
            ray_directions=[pose_ray_dict[link]['vector'] for link in pose_ray_dict.keys()],
            multiple_hits=True,
            max_hits=10,
            return_locations=True
        )

        surface_norm = self.obj.generate_geometry(frame_idx).face_normals[index_tri]
        return location, surface_norm


    def check_frame_exist(self, frame_idx):
        ''' If frame index is not present in one of the dictionaries, return false'''
        return self.animal.check_frame_exist(frame_idx) and self.obj.check_frame_exist(frame_idx)

