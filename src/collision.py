import logging
import numpy as np
import trimesh

# import trimesh.viewer
from src.animal import AnimalStruct
from src.object import CollisionObj

logger = logging.getLogger(__name__)


class CollisionDetector:

    def __init__(self, obj_folder: str, skeleton_json_path: str, pose_csv: str):
        self._obj_folder = obj_folder
        link_names = ['thorax', 'neck', 'a_L1', 'a_L2', 'a_R1', 'a_R2', 'leg_f_L1', 'leg_f_L2', 'leg_f_R1', 'leg_f_R2', 'leg_m_L1', 'leg_m_L2', 'leg_m_R1', 'leg_m_R2', 'a_R0', 'a_L0', 'leg_f_R0', 'leg_f_L0', 'leg_m_R0', 'leg_m_L0', 'eye_L', 'eye_R', 'm_L0', 'm_R0', 'm_L1', 'm_R1']
        skeleton_connectivity = [(0, 1), (0, 17), (0, 16), (0, 19), (0, 18), (1, 14), (1, 15), (1, 22), (1, 23), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 4), (15, 2), (16, 8), (17, 6), (18, 12), (19, 10), (20, 1), (21, 1), (22, 24), (23, 25)]
        self.animal = AnimalStruct(skeleton_connectivity, link_names, pose_csv)
        self.obj = CollisionObj(self._obj_folder)

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
            logger.warning('Could not find collision for frame index {}'.format(frame_idx))
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

