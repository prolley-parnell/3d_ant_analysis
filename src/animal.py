import logging
import numpy as np
import trimesh

import trimesh.transformations as tf
import trimesh.viewer

from scipy.sparse.csgraph import dijkstra
from typing import List, Tuple

from src.skeleton import SkeletonToml

logger = logging.getLogger(__name__)

class AnimalStruct:

    def __init__(self,
                 toml_path: str,
                 pose_csv: str,
                 units: str = "mm"
                 ):
        ''' Initialise a holder for the structure of the animal skeleton
        :param connectivity_list: List of tuples containing pairs of nodes with a connection in the skeleton
        :param node_name_list: List of node names, the order of this list corresponds to the numbering of nodes in the connectivity_list
        :param pose_csv: Path to the csv file containing the pose of the skeleton
        '''
        sk = SkeletonToml(toml_path)
        self._connectivity_list = sk.skeleton_connectivity
        self.node_name_list = sk.link_name_list
        self._connectivity_dict = {}
        self._units = units
        self._generate_connectivity_dict()
        self._pose_array = np.genfromtxt(pose_csv, delimiter=',', names=True, filling_values=np.nan, dtype=np.float64)
        self._pose_dict = self._pose_csv_to_dict(self._pose_array, self.node_name_list)
        self._pose_ray_dict = {} # A dictionary containing the rays between parent nodes and children



    @staticmethod
    def _pose_csv_to_dict(pose_array, link_names: list[str]):
        ''' Assumes the structure is _x,_y, _z, _score'''

        pose_dict = {}
        for frame_i, row in enumerate(pose_array):
            if row.dtype.names.__contains__('frame'):
                frame_i = int(row['frame'])
            else:
                logging.debug("Frame column not included in the CSV")


            temp_dict = {}
            all_nan_flag = True
            for name in link_names:
                temp_dict[name] = {}
                temp_dict[name]['xyz'] = np.array(
                    [row[name + "_x"], row[name + "_y"], row[name + "_z"]])
                temp_dict[name]['score'] = row[name + "_score"]
                if temp_dict[name]['score'] > 0:
                    all_nan_flag = False

            if not all_nan_flag:
                pose_dict[frame_i] = temp_dict

        return pose_dict

    def _generate_connectivity_dict(self, reference_index=0):
        ''' Create a dictionary object that can be used to reference nodes by name to find their connectivity, and distance to the base node'''

        distance_list, connectivity_map = self._tree_search_connectivity(reference_index)

        for i, name in enumerate(self.node_name_list):
            connections = list(np.where(connectivity_map[i])[0])
            if i == reference_index:
                parent = None
            else:
                parent_index = np.argmin(distance_list[connections])
                parent = self.node_name_list[connections[parent_index]]
            self._connectivity_dict[name] = {'dist': distance_list[i],
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

    def _generate_rays(self, frame_idx):
        ''' For the skeleton at the frame index provided, generate a series of origins and vector axes to represent the
         links between the key nodes'''
        ray_dict = {}

        # Find all links in skeleton and assign names
        for node in self._connectivity_dict:
            parent_name = self._connectivity_dict[node]['parent']
            if parent_name is not None and self._pose_dict[frame_idx][node]['score'] > 0.003:

                # Assign the direction based on distance to core/anchor node
                point_a = self._pose_dict[frame_idx][parent_name]['xyz']
                point_b = self._pose_dict[frame_idx][node]['xyz']

                if any(np.isnan(point_a)) or any(np.isnan(point_b)):
                    logging.info("Point is NaN")
                else:
                    # Generate the vector and origin based on this direction
                    origin = point_a
                    vector = point_b - point_a
                    destination = point_b

                    # Assign vector and origin to key name
                    name = parent_name + "_to_" + node
                    ray_dict[name] = {}

                    ray_dict[name]['origin'] = origin
                    ray_dict[name]['vector'] = vector
                    ray_dict[name]['dest'] = destination

        return ray_dict

    def check_frame_exist(self, frame_idx):
        ''' If frame index is not present in the pose dictionary, return false'''
        if not self._pose_dict.keys().__contains__(frame_idx):
            logger.debug('Frame index {} not present in the animal skeleton dictionary \n'.format(frame_idx))
            return False
        return True

    def get_pose_ray(self, frame_idx: int):
        '''
        Returns a dictionary object with the following structure:
        dict[ray_name: String with naming convention "nodeA_to_nodeB"]{
        origin: Origin of the ray
        vector: Vector of origin to destination
        dest: Destination of the ray
        }
        '''
        if not self._pose_ray_dict.keys().__contains__(frame_idx):
            if self.check_frame_exist(frame_idx):
                self._pose_ray_dict[frame_idx] = self._generate_rays(frame_idx)

        return self._pose_ray_dict[frame_idx]



    def generate_scene(self, frame_idx: int):
        ''' Return the trimesh scene for the animal skeleton for the given frame index
        :returns: A scene where the geometry is represented by a pointcloud and a set of paths defined by the
                connecting edges of the skeleton
        '''

        animal_geometry = self.generate_geometry(frame_idx)
        if animal_geometry is not None:
            animal_ray, animal_node = animal_geometry
            scene = trimesh.Scene([animal_ray, animal_node])
        else:
            # create a scene containing no geometry
            scene = trimesh.Scene()
            scene.add_geometry(None)

        return scene

    def generate_geometry(self, frame_idx: int):
        ''' Return the geometry objects for the animal skeleton for the given frame index
        :returns: A trimesh object geometry where the geometry is represented by a pointcloud and a set of paths defined by the
                connecting edges of the skeleton
        '''
        if self.check_frame_exist(frame_idx):
            pose_ray_dict = self.get_pose_ray(frame_idx)

            if len(pose_ray_dict) == 0:
                logger.error("No rays present for animal at frame {}".format(frame_idx))
                return None

            ray_origins = np.array([pose_ray_dict[link]['origin'] for link in pose_ray_dict.keys()])
            ray_destination = np.array([pose_ray_dict[link]['dest'] for link in pose_ray_dict.keys()])
            ray_visualise = trimesh.load_path(
                np.hstack((ray_origins, ray_destination)).reshape(-1, 2, 3)
            )

            nodes = trimesh.points.PointCloud(ray_origins)

            #Set the units
            ray_visualise.units = self._units
            nodes.units = self._units

            # create a unique color for each point
            cloud_colors = np.array([trimesh.visual.random_color() for i in nodes])
            # set the colors on the random point and its nearest point to be the same
            nodes.vertices_color = cloud_colors

        else:
            return None

        return ray_visualise, nodes

    def get_frame_range(self):
        ''' Give [min, max] of the frame indices for the stored poses'''
        return [min(self._pose_dict.keys()), max(self._pose_dict.keys())]

    def get_animal_pose(self, frame_idx: int):
        '''
        Return a dictionary that contains the name of the joint in the csv and the transform relative to the world pose at the given frame index
        :param frame_idx: frame index query
        :type frame_idx: int
        :return: dict if frame is present, otherwise None
        '''

        if self.check_frame_exist(frame_idx):
            return {name: data['xyz'] for name, data in self._pose_dict[frame_idx].items()}
        return {}

    def initialise_scene(self, frame_idx: int):

        animal_pose = self.get_animal_pose(frame_idx)
        scene = trimesh.Scene()

        if animal_pose is not None:
            for name, pose in animal_pose.items():
                if not np.isnan(pose).any():
                    # sphere
                    geom = trimesh.creation.icosphere(radius=0.05)
                    geom.visual.face_colors = np.random.uniform(0, 1, (len(geom.faces), 3))
                    transform = tf.translation_matrix(pose)
                    scene.add_geometry(geom, transform=transform, node_name=name, geom_name=name)
        else:
            scene.add_geometry(None)
        return scene


