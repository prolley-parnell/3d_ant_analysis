import logging
import numpy as np
import trimesh
from pathlib import Path
import concurrent.futures
import trimesh.transformations as tf
import trimesh.viewer
from typing import Optional

from scipy.sparse.csgraph import dijkstra

from src.skeleton import SkeletonToml
from pandas import DataFrame, MultiIndex

logger = logging.getLogger(__name__)

class AnimalStruct:

    def __init__(self,
                 toml_path: Path,
                 pose_csv: Path,
                 input_units: str = "mm",
                 animal_name: str = None
                 ):
        """ Initialise a holder for the structure of the animal skeleton
        :param toml_path: Path to the toml file containing the structure of the animal skeleton
        :param pose_csv: Path to the csv file containing the pose of the skeleton
        """
        sk = SkeletonToml(toml_path)
        self._connectivity_list = sk.skeleton_connectivity
        self._node_name_list = sk.link_name_list

        if pose_csv is not Path:
            pose_csv = Path(pose_csv).resolve()

        if animal_name is None:
            #Structure is 240905-1616_session28_track13_points3d.csv
            self.name = pose_csv.as_posix().split("/")[-1].split("_")[2]
            self._prefix = pose_csv.as_posix().split("/")[-1].split("_")[0]
            self._session = pose_csv.as_posix().split("/")[-1].split("_")[1]
        else:
            self.name = animal_name
        self._connectivity_dict = {}
        self._units = input_units
        self._generate_connectivity_dict()
        self._pose_array = np.genfromtxt(pose_csv, delimiter=',', names=True, filling_values=np.nan, dtype=np.float64)
        self._pose_dict = self._pose_csv_to_dict(self._pose_array, self._node_name_list)
        self._pose_ray_dict = {} # A dictionary containing the rays between parent nodes and children
        self._colour = trimesh.visual.random_color()
        self._ray_names = {}


    @staticmethod
    def _pose_csv_to_dict(pose_array, link_names: list[str]) -> dict:
        """ Assumes the structure is _x,_y, _z, _score"""

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

                if not np.any(np.isnan(temp_dict[name]['xyz'])):
                    all_nan_flag = False

            if not all_nan_flag:
                pose_dict[frame_i] = temp_dict

        if pose_dict == {}:
            raise ValueError("Pose csv does not contain any valid frames")

        return pose_dict

    def _generate_connectivity_dict(self, reference_index=0) -> None:
        """ Create a dictionary object that can be used to reference nodes by name to find their connectivity, and distance to the base node"""

        distance_list, connectivity_map = self._tree_search_connectivity(reference_index)

        for i, name in enumerate(self._node_name_list):
            connections = list(np.where(connectivity_map[i])[0])
            if i == reference_index:
                parent = None
            else:
                parent_index = np.argmin(distance_list[connections])
                parent = self._node_name_list[connections[parent_index]]
            self._connectivity_dict[name] = {'dist': distance_list[i],
                                            'index': i,
                                            'connections': connections,
                                            'parent': parent }

    def _tree_search_connectivity(self, reference_index: int):
        """ Use a connectivity list ([(node_1, node_2), (node_3, node_2)])  and find number of steps to reach reference
        index. Returns a list of distances per node"""

        n_node = len(np.unique(self._connectivity_list))

        connectivity_map = np.zeros((n_node, n_node))
        for (a, b) in self._connectivity_list:
            connectivity_map[a, b] = 1
            connectivity_map[b, a] = 1

        dist_list = dijkstra(csgraph=connectivity_map, directed=False, indices=reference_index, return_predecessors=False)

        return dist_list, connectivity_map

    def _generate_rays(self, frame_idx) -> dict:
        """ For the skeleton at the frame index provided, generate a series of origins and vector axes to represent the
         links between the key nodes"""
        ray_dict = {}

        # Find all links in skeleton and assign names
        for node in self._connectivity_dict:
            parent_name = self._connectivity_dict[node]['parent']

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
                    destination = point_b

                    # Assign vector and origin to key name
                    name = parent_name + "_to_" + node
                    if name not in self._ray_names:
                        self._ray_names[name] = {'parent' : parent_name, 'child' : node}

                    ray_dict[name] = {}

                    ray_dict[name]['origin'] = origin
                    ray_dict[name]['vector'] = vector
                    ray_dict[name]['dest'] = destination

        return ray_dict

    def check_frame_exist(self, frame_idx) -> bool:
        """ If frame index is not present in the pose dictionary, return false"""
        if not self._pose_dict.keys().__contains__(frame_idx):
            logger.debug('Frame index {} not present in the animal skeleton dictionary \n'.format(frame_idx))
            return False
        return True

    def get_pose_ray(self, frame_idx: int) -> dict:
        """
        Returns a dictionary object with the following structure:
        dict[ray_name: String with naming convention "nodeA_to_nodeB"]{
        origin: Origin of the ray
        vector: Vector of origin to destination
        dest: Destination of the ray
        }
        """
        if self.check_frame_exist(frame_idx):
            if not self._pose_ray_dict.keys().__contains__(frame_idx):
                self._pose_ray_dict[frame_idx] = self._generate_rays(frame_idx)

            return self._pose_ray_dict[frame_idx]
        else:
            return {}



    def generate_scene(self, frame_idx: int) -> trimesh.scene.Scene:
        """ Return the trimesh scene for the animal skeleton for the given frame index
        :returns: A scene where the geometry is represented by a pointcloud and a set of paths defined by the
                connecting edges of the skeleton
        """

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
        """ Return the geometry objects for the animal skeleton for the given frame index
        :returns: A trimesh object geometry where the geometry is represented by a pointcloud and a set of paths defined by the
                connecting edges of the skeleton
        """
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
            cloud_colors = np.array([self._colour for i in nodes.vertices])
            # set the colors on the random point and its nearest point to be the same
            nodes.vertices_color = cloud_colors

            ray_colours = np.array([self._colour for i in ray_visualise.entities])
            ray_visualise.colors = ray_colours

        else:
            return None

        return ray_visualise, nodes

    def get_frame_range(self) -> list:
        """ Give [min, max] of the frame indices for the stored poses"""
        return [min(self._pose_dict.keys()), max(self._pose_dict.keys())]

    def get_animal_pose(self, frame_idx: int) -> dict:
        """
        Return a dictionary that contains the name of the joint in the csv and the transform relative to the world pose at the given frame index
        :param frame_idx: frame number to query
        :type frame_idx: int
        :return: dict if frame is present, otherwise empty dict
        """

        if self.check_frame_exist(frame_idx):
            return {name: data['xyz'] for name, data in self._pose_dict[frame_idx].items()}
        return {}

    def initialise_scene(self, frame_idx: int) -> trimesh.scene.Scene:

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

    @property
    def colour(self):
        return self._colour

    @property
    def frames(self) -> list:
        return [*self._pose_dict.keys()]

    @property
    def ray_names(self) -> dict:
        return self._ray_names

    def get_xyz_df(self, frame_idx: list = None, node_list: list = None):
        if frame_idx is None:
            frame_idx = self.frames

        frame_idx = sorted(frame_idx)

        if node_list is None:
            node_list = self._node_name_list

        m = MultiIndex.from_product([node_list, ['x', 'y', 'z']], names=['Node', 'Axis'])
        df = DataFrame(None, index=m, columns=frame_idx)

        for frame in frame_idx:
            for name in node_list:
                if frame in self._pose_dict.keys():
                    df.loc[name, frame] = self._pose_dict[frame][name]['xyz'].copy()
                else:
                    df.loc[name, frame] = np.nan

        return df, node_list

    def update_from_df(self, dataframe_in: DataFrame):
        def df_to_dict(dataframe: DataFrame):
            pose_dict = {}
            kp_in_df = dataframe.index.droplevel(level=1).unique()

            for frame in dataframe:
                temp_dict = {}
                all_nan_flag = True
                for name in kp_in_df:
                    temp_dict[name] = {}
                    temp_dict[name]['xyz'] = np.array(np.array(dataframe.loc[name, frame], dtype=np.float64))
                    temp_dict[name]['score'] = 1

                    if not np.any(np.isnan(temp_dict[name]['xyz'])):
                        all_nan_flag = False
                if not all_nan_flag:
                    pose_dict[frame] = temp_dict
            if pose_dict == {}:
                raise ValueError("Pose csv does not contain any valid frames")

            return pose_dict

        new_pose_dict = df_to_dict(dataframe_in)
        self._pose_dict = new_pose_dict
        self._pose_ray_dict = {}



class AnimalList:
    def __init__(self,
        animals: Optional[list[AnimalStruct]] = None,
        toml_path: Optional[Path | str] = None,
        csv_folder: Optional[Path | str] = None,
        session_number: Optional[int] = None,
        track_number: Optional[list] = None,
        input_units: Optional[str] = "mm"
        ):
        """ Generate a list of AnimalStruct objects from a folder structure"""
        if animals is None:
            self._units = input_units
            self._toml_path = toml_path

            if csv_folder is not Path:
                csv_folder = Path(csv_folder).resolve()

            #240905-1616_session28_track13_points3d.csv
            if track_number is None:
                csv_path_list = sorted(csv_folder.glob(f"*_session{str(session_number)}_track*_points3d.csv"))
            else:
                csv_path_list = []
                for track_id in track_number:
                    csv_path_list.append(sorted(csv_folder.glob(f"*_session{str(session_number)}_track{str(track_id)}_points3d.csv")))

            self._animals = []
            self._read_tracking_folder_mt(csv_path_list)
        else:
            self._animals = animals

        self._animal_name_list = [animal.name for animal in self._animals]


    def _read_tracking_folder_mt(self, tracking_dir_list: list):
        """ Given a path to a folder containing ".obj" or ".dae" files with the name of the corresponding frame, load these and
        convert to a dict of trimesh"""

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_path = {executor.submit(AnimalStruct, self._toml_path, track_path, self._units): track_path for track_path in tracking_dir_list}
            for future in concurrent.futures.as_completed(future_to_path):
                path_input = future_to_path[future]
                try:
                    animal = future.result()
                except Exception as exc:
                    logger.error('%r generated an exception: %s' % (path_input, exc))
                else:
                    logger.info('Animal name is %s ' % animal.name)
                    self._animals.append(animal)

    @property
    def animals(self) -> list[AnimalStruct]:
        return self._animals

    def animal(self, animal_name: str) -> AnimalStruct | None:
        for animal in self._animals:
            if animal.name == animal_name:
                return animal
        return None

    @property
    def animal_name_list(self):
        return self._animal_name_list

    def animal_colour(self, animal_name: Optional[str] = None) -> [np.ndarray]:
        if animal_name is None:
            colour = [animal.colour for animal in self._animals]
        else:
            colour = [animal.colour for animal in self._animals if animal.name in animal_name]
        return colour

    def where_frame_exist(self, frame_idx: int) -> list[AnimalStruct]:
        """ :returns A list of animals where the animal is in the frame"""
        return [animal for animal in self._animals if animal.check_frame_exist(frame_idx)]

    def pre_load_ray(self):
        """ Calculate all rays at once for all animals in the list"""
        for animal in self._animals:
            frames = animal.get_frame_range()
            for frame in range(*frames):
                animal.get_pose_ray(frame)
