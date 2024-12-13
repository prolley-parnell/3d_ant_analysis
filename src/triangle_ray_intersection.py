import logging
import numpy as np
import concurrent.futures
import trimesh
import os

from jedi.inference.gradual.typing import Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from typing import List, Tuple

from scipy.spatial import distance_matrix

logger = logging.getLogger(__name__)


class AnimalStruct:

    def __init__(self,
                 connectivity_list: List[Tuple[int, int]],
                 node_name_list: List[str]
                 ):
        ''' Initialise a holder for the structure of the animal skeleton
        :parameter:
        connectivity_list: List of tuples containing pairs of nodes with a connection in the skeleton
        node_name_list: List of node names, the order of this list corresponds to the numbering of nodes in the connectivity_list

        '''
        self._connectivity_list = connectivity_list
        self.node_name_list = node_name_list
        self.connectivity_dict = {}
        self.generate_connectivity_dict()

    def generate_connectivity_dict(self, reference_index=0):
        ''' Create a dictionary object that can be used to reference nodes by name to find their connectivity, and distance to the base node'''

        distance_list, connectivity_map = self._tree_search_connectivity(reference_index)

        for i, name in enumerate(self.node_name_list):
            connections = list(np.where(connectivity_map[i])[0])
            if i == reference_index:
                parent = None
            else:
                parent_index = np.argmin(distance_list[connections])
                parent = self.node_name_list[connections[parent_index]]
            self.connectivity_dict[name] = {'dist': distance_list[i],
                                            'index': i,
                                            'connections': connections,
                                            'parent': parent }

    def _tree_search_connectivity(self, reference_index: int):
        ''' Use a connectivity list ([(node_1, node_2), (node_3, node_2)])  and find number of steps to reach reference
        index. Returns a list of distances per node'''

        n_node = len(np.unique(self._connectivity_list))

        connectivity_map = np.zeros((n_node, n_node))
        for (a, b) in self._connectivity_list:
            connectivity_map[a, b] = 1
            connectivity_map[b, a] = 1

        dist_list = dijkstra(csgraph=connectivity_map, directed=False, indices=reference_index, return_predecessors=False)

        return dist_list, connectivity_map

class CollisionDetector:

    def __init__(self, obj_folder: str, skeleton_json_path: str, pose_csv: str):
        self._obj_folder = obj_folder
        self._obj_dict = {}
        self._read_obj_folder_mt(self._obj_folder)
        link_names = link_names = ['thorax', 'neck', 'a_L1', 'a_L2', 'a_R1', 'a_R2', 'leg_f_L1', 'leg_f_L2', 'leg_f_R1', 'leg_f_R2', 'leg_m_L1', 'leg_m_L2', 'leg_m_R1', 'leg_m_R2', 'a_R0', 'a_L0', 'leg_f_R0', 'leg_f_L0', 'leg_m_R0', 'leg_m_L0', 'eye_L', 'eye_R', 'm_L0', 'm_R0', 'm_L1', 'm_R1']
        skeleton_connectivity = [(0, 1), (0, 17), (0, 16), (0, 19), (0, 18), (1, 14), (1, 15), (1, 22), (1, 23), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 4), (15, 2), (16, 8), (17, 6), (18, 12), (19, 10), (20, 1), (21, 1), (22, 24), (23, 25)]
        self._animal = AnimalStruct(skeleton_connectivity, link_names)
        #self._distance_map = self._tree_search_connectivity(self._skeleton_connectivity, 0)
        self._pose_array = np.genfromtxt(pose_csv, delimiter=',', names=True, filling_values=np.nan, dtype=np.float64)
        self._pose_dict = self._pose_csv_to_dict(self._pose_array, self._animal.node_name_list)
        self._pose_ray_dict = {}



    def _read_obj_folder(self, obj_dir_path: str):
        ''' Given a path to a folder containing ".obj" files with the name of the corresponding frame, load these and
        convert to a dict of trimesh'''
        # scan all the OBJ file names in this directory
        path_list = os.listdir(obj_dir_path)

        for p in path_list:
            frame_idx, trimesh_obj = self._single_obj_to_trimesh(p)
            self._obj_dict[frame_idx] = trimesh_obj


    def _read_obj_folder_mt(self, obj_dir_path: str):
        ''' Given a path to a folder containing ".obj" files with the name of the corresponding frame, load these and
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
                    self._obj_dict[frame_idx] = trimesh_obj

    @staticmethod
    def _pose_csv_to_dict(pose_array, link_names: List[str]):
        ''' Assumes the structure is _x,_y, _z, _score'''

        pose_dict = {}
        for row in pose_array:
            pose_dict[int(row['frame'])] = {}
            for name in link_names:
                pose_dict[int(row['frame'])][name] = {}
                # x_mask = [ name+"_x" in csv_name for csv_name in csv_col]
                # y_mask = [ name+"_y" in csv_name for csv_name in csv_col]
                # z_mask = [ name+"_z" in csv_name for csv_name in csv_col]
                pose_dict[int(row['frame'])][name]['xyz'] = np.array([row[name+"_x"],row[name+"_y"],row[name+"_z"]])
                # score_mask = [name + "_score" in csv_name for csv_name in csv_col]
                pose_dict[int(row['frame'])][name]['score'] = row[name+"_score"]

        return pose_dict

    def _sleap_json_to_dict(self, skeleton_file: str):
        ''' Given path to JSON file describing skeleton, convert to dict with connectivity'''


    def _single_obj_to_trimesh(self, obj_path: str):
        ''' Convert the obj at the given path to a trimesh object and return the object and frame index '''
        if not obj_path.startswith('.') and os.path.splitext(obj_path)[-1] in [".obj", ".OBJ"]:
            frame_index = int(os.path.splitext(obj_path)[-2].split('_')[-1]) # Add to only get the number next to the obj extension
            return frame_index, trimesh.load(self._obj_folder + obj_path, force='mesh')


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
        scene = trimesh.Scene([self._obj_dict[frame_idx], ax, ray_visualise])
        # show the visualisation
        return scene

    def visualise_animal(self, frame_idx: int):
        if not self._pose_ray_dict.keys().__contains__(frame_idx):
            self._pose_ray_dict[frame_idx] = self._generate_rays(frame_idx)

        pose_ray_dict = self._pose_ray_dict[frame_idx]
        ray_origins = np.array([pose_ray_dict[link]['origin'] for link in pose_ray_dict.keys()])
        ray_directions = np.array([pose_ray_dict[link]['vector'] for link in pose_ray_dict.keys()])
        ray_visualise = trimesh.load_path(
            np.hstack((ray_origins, ray_origins + ray_directions)).reshape(-1, 2, 3)
        )
        nodes = trimesh.points.PointCloud(ray_origins)

        # create a unique color for each point
        cloud_colors = np.array([trimesh.visual.random_color() for i in nodes])

        # set the colors on the random point and its nearest point to be the same
        nodes.vertices_color = cloud_colors

        #Make some axes to indicate direction
        ax = trimesh.creation.axis(10)

        # create a scene containing the mesh and two sets of points
        scene = trimesh.Scene([ray_visualise, ax, nodes])
        return scene

    def get_collisions(self, frame_idx: int):
        '''
        Given a frame index, calculate and return the collisions with the object and the surface normal at the
        point of collision.
        '''
        if not self._check_frame_exist(frame_idx):
            logger.warning('Could not find collision for frame index {}'.format(frame_idx))
            return None

        if not self._pose_ray_dict.keys().__contains__(frame_idx):
            self._pose_ray_dict[frame_idx] = self._generate_rays(frame_idx)

        pose_ray_dict = self._pose_ray_dict[frame_idx]
        #ri = trimesh.ray.ray_pyembree.RayMeshIntersector(self._obj_dict[frame_idx])
        ri = trimesh.ray.ray_triangle.RayMeshIntersector(self._obj_dict[frame_idx])

        index_tri, index_ray, location = ri.intersects_id(
            ray_origins=[pose_ray_dict[link]['origin'] for link in pose_ray_dict.keys()],
            ray_directions=[pose_ray_dict[link]['vector'] for link in pose_ray_dict.keys()],
            multiple_hits=True,
            max_hits=10,
            return_locations=True
        )

        surface_norm = self._obj_dict[frame_idx].face_normals[index_tri]
        return location, surface_norm


    def _check_frame_exist(self, frame_idx):
        ''' If frame index is not present in one of the dictionaries, return false'''
        if not self._obj_dict.keys().__contains__(frame_idx):
            logger.debug('Frame index {} not present in the object mesh dictionary \n'.format(frame_idx))
            return False
        if not self._pose_dict.keys().__contains__(frame_idx):
            logger.debug('Frame index {} not present in the animal skeleton dictionary \n'.format(frame_idx))
            return False
        return True


    def _generate_rays(self, frame_idx):
        ''' For the skeleton at the frame index provided, generate a series of origins and vector axes to represent the
         links between the key nodes'''
        ray_dict = {}

        # Find all links in skeleton and assign names
        for node in self._animal.connectivity_dict:
            parent_name = self._animal.connectivity_dict[node]['parent']
            if parent_name is not None:

                # Assign the direction based on distance to core/anchor node
                point_a = self._pose_dict[frame_idx][parent_name]['xyz']
                point_b = self._pose_dict[frame_idx][node]['xyz']

                if any(np.isnan(point_a)) or any(np.isnan(point_b)):
                    logging.info("Point is NaN")
                else:
                    # Generate the vector and origin based on this direction
                    origin = point_a
                    vector = point_b - point_a

                    # Assign vector and origin to key name
                    name = parent_name + "_to_" + node
                    ray_dict[name] = {}

                    ray_dict[name]['origin'] = origin
                    ray_dict[name]['vector'] = vector

        return ray_dict

