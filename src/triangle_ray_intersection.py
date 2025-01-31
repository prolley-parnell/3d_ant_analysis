import logging
import numpy as np
import concurrent.futures
import trimesh
import os
import glooey

import pyglet
import trimesh.transformations as tf
import trimesh.viewer


from scipy.sparse.csgraph import dijkstra
from typing import List, Tuple

logger = logging.getLogger(__name__)


class AnimalStruct:

    def __init__(self,
                 connectivity_list: List[Tuple[int, int]],
                 node_name_list: List[str],
                 pose_csv: str
                 ):
        ''' Initialise a holder for the structure of the animal skeleton
        :param connectivity_list: List of tuples containing pairs of nodes with a connection in the skeleton
        :param node_name_list: List of node names, the order of this list corresponds to the numbering of nodes in the connectivity_list
        :param pose_csv: Path to the csv file containing the pose of the skeleton
        '''
        self._connectivity_list = connectivity_list
        self.node_name_list = node_name_list
        self._connectivity_dict = {}
        self._generate_connectivity_dict()
        self._pose_array = np.genfromtxt(pose_csv, delimiter=',', names=True, filling_values=np.nan, dtype=np.float64)
        self._pose_dict = self._pose_csv_to_dict(self._pose_array, self.node_name_list)
        self._pose_ray_dict = {} # A dictionary containing the rays between parent nodes and children



    @staticmethod
    def _pose_csv_to_dict(pose_array, link_names: List[str]):
        ''' Assumes the structure is _x,_y, _z, _score'''

        pose_dict = {}
        for row in pose_array:
            pose_dict[int(row['frame'])] = {}
            for name in link_names:
                pose_dict[int(row['frame'])][name] = {}
                pose_dict[int(row['frame'])][name]['xyz'] = np.array(
                    [row[name + "_x"], row[name + "_y"], row[name + "_z"]])
                pose_dict[int(row['frame'])][name]['score'] = row[name + "_score"]

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
            self._pose_ray_dict[frame_idx] = self._generate_rays(frame_idx)
        return self._pose_ray_dict[frame_idx]


    def visualise_animal(self, frame_idx: int):
        ''' Return the trimesh scene for the animal skeleton for the given frame index
        :returns: A scene where the geometry is represented by a pointcloud and a set of paths defined by the
                connecting edges of the skeleton
        '''
        if self.check_frame_exist(frame_idx):
            pose_ray_dict = self.get_pose_ray(frame_idx)
            ray_origins = np.array([pose_ray_dict[link]['origin'] for link in pose_ray_dict.keys()])
            ray_destination = np.array([pose_ray_dict[link]['dest'] for link in pose_ray_dict.keys()])
            ray_visualise = trimesh.load_path(
                np.hstack((ray_origins, ray_destination)).reshape(-1, 2, 3)
            )
            if not ray_origins.any():
                scene = trimesh.Scene()
                scene.add_geometry(None)
                return scene

            nodes = trimesh.points.PointCloud(ray_origins)

            # create a unique color for each point
            cloud_colors = np.array([trimesh.visual.random_color() for i in nodes])

            # set the colors on the random point and its nearest point to be the same
            nodes.vertices_color = cloud_colors
            # create a scene containing the mesh and two sets of points
            scene = trimesh.Scene([ray_visualise, nodes])
        else:
            # create a scene containing no geometry
            scene = trimesh.Scene()
            scene.add_geometry(None)

        return scene

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
        return None

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




class CollisionObj:

    def __init__(self, obj_folder: str):
        '''
        Initialise an object given by a folder of "obj" files into a dictionary of trimesh objects.
        :param obj_folder: Path to the folder containing the objects
        '''
        self._obj_folder = obj_folder
        self._obj_dict = {}
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
                    print(os.path.isdir(self._obj_folder + path_input))
                    logger.error('%r generated an exception: %s' % (path_input, exc))
                else:
                    logger.info('Frame index is %d ' % frame_idx)
                    self._obj_dict[frame_idx] = trimesh_obj


    def _single_obj_to_trimesh(self, obj_path: str):
        ''' Convert the obj at the given path to a trimesh object and return the object and frame index '''
        if os.path.isdir(self._obj_folder + obj_path):
            raise Exception("Folder {} ignored".format(self._obj_folder + obj_path))
        if os.path.splitext(obj_path)[-1] in [".obj", ".OBJ"] and not obj_path.startswith('.'):
            frame_index = int(os.path.splitext(obj_path)[-2].split('_')[-1]) # Add to only get the number next to the obj extension
            return frame_index, trimesh.load(self._obj_folder + obj_path, force='mesh')

    def get_obj_trimesh(self, frame_idx: int):
        '''Return a trimesh of the object at the given frame index'''
        if self.check_frame_exist(frame_idx):
            return self._obj_dict[frame_idx]
        else:
            return None

    def generate_obj_scene(self, frame_idx: int):
        '''Return a trimesh scene object with the seed in the world frame'''
        scene = trimesh.Scene()
        scene.add_geometry(self.get_obj_trimesh(frame_idx), node_name="object", geom_name=str(frame_idx))
        return scene




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
        scene = trimesh.Scene([self.obj.get_obj_trimesh(frame_idx), ax, ray_visualise])

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
        ri = trimesh.ray.ray_triangle.RayMeshIntersector(self.obj.get_obj_trimesh(frame_idx))

        index_tri, index_ray, location = ri.intersects_id(
            ray_origins=[pose_ray_dict[link]['origin'] for link in pose_ray_dict.keys()],
            ray_directions=[pose_ray_dict[link]['vector'] for link in pose_ray_dict.keys()],
            multiple_hits=True,
            max_hits=10,
            return_locations=True
        )

        surface_norm = self.obj.get_obj_trimesh(frame_idx).face_normals[index_tri]
        return location, surface_norm


    def check_frame_exist(self, frame_idx):
        ''' If frame index is not present in one of the dictionaries, return false'''
        return self.animal.check_frame_exist(frame_idx) and self.obj.check_frame_exist(frame_idx)


class Viewer:

    """
    Example application that includes moving camera, scene and image update.
    """

    def __init__(self, animal, object, frame_index):
        # create window with padding
        self.width, self.height = 480 * 2, 360
        window = self._create_window(width=self.width, height=self.height)

        gui = glooey.Gui(window)

        hbox = glooey.HBox()
        hbox.set_padding(5)

        # scene widget for changing camera location
        self.object = object
        self.animal = animal
        self.frame_range = self.animal.get_frame_range()
        self.frame_index = max(frame_index, self.frame_range[0])

        obj_scene = self.object.generate_obj_scene(frame_idx=self.frame_index)

        self.scene_widget_obj = trimesh.viewer.SceneWidget(obj_scene)
        hbox.add(self.scene_widget_obj)

        # scene widget for changing scene
        animal_scene = self.animal.visualise_animal(frame_idx=self.frame_index)
        geom = trimesh.path.creation.box_outline((0.6, 0.6, 0.6))
        animal_scene.add_geometry(geom)
        self.scene_widget_animal = trimesh.viewer.SceneWidget(animal_scene)
        self.scene_widget_animal._angles = [np.deg2rad(45), 0, 0]
        hbox.add(self.scene_widget_animal)

        # scene_comb = trimesh.Scene()
        # scene_comb.add_geometry()
        # self.scene_widget_combined = trimesh.viewer.SceneWidget(scene_comb)
        # hbox.add(self.scene_widget_combined)

        gui.add(hbox)

        pyglet.clock.schedule_interval(self.callback, 1.0 / 10)
        pyglet.app.run()

    def callback(self, dt):
        #Update the frame counter
        self.frame_index = (self.frame_index + 1) % self.frame_range[1]

        #Check if the frame is present in the dict of animal frames
        if self.animal.check_frame_exist(self.frame_index):
            #Remove existing geometry from the scene
            self.scene_widget_animal.do_undraw()
            #Replace the current scene with the scene created in the animal class
            self.scene_widget_animal.scene = self.animal.visualise_animal(frame_idx=self.frame_index)

            self.scene_widget_animal.scene.set_camera()
            #Redraw the new scene
            self.scene_widget_animal.do_draw()
            #todo: set the camera so it covers the approximate area and does not change size or distance to animal if nodes disappear


        #Check if the frame is present in the dict of object frames
        if self.object.check_frame_exist(self.frame_index):
            #Remove existing geometry from the scene
            self.scene_widget_obj.do_undraw()
            #Get the geometry from the dict in the obj class
            geom = self.object.get_obj_trimesh(self.frame_index)
            self.scene_widget_obj.scene.add_geometry(geom, node_name="object", geom_name=str(self.frame_index))
            #Redraw the new scene
            self.scene_widget_obj.do_draw()

        # scene_comb = self.scene_widget_obj.scene.append_scenes(self.scene_widget_animal.scene)
        # self.scene_widget_combined.scene = scene_comb
        # self.scene_widget_combined.do_draw()

    def _create_window(self, width, height):
        try:
            config = pyglet.gl.Config(
                sample_buffers=1, samples=4, depth_size=24, double_buffer=True
            )
            window = pyglet.window.Window(config=config, width=width, height=height)
        except pyglet.window.NoSuchConfigException:
            config = pyglet.gl.Config(double_buffer=True)
            window = pyglet.window.Window(config=config, width=width, height=height)

        @window.event
        def on_key_press(symbol, modifiers):
            if modifiers == 0:
                if symbol == pyglet.window.key.Q:
                    window.close()

        return window
