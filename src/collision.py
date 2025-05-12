import logging
from typing import Optional

import numpy as np
import trimesh

from src.animal import AnimalStruct, AnimalList
from src.object import CollisionObj, CollisionObjTransform
from src.loader import InstanceLoader

from pandas import DataFrame

import concurrent.futures


logger = logging.getLogger(__name__)


class CollisionDetector:

    def __init__(self, instance: Optional[InstanceLoader] = None,
                 animal_list: Optional[AnimalList] = None,
                 obj_list: Optional[list[CollisionObj | CollisionObjTransform | trimesh.Trimesh]] = None,
                 node_of_interest: Optional[list[str]] = None):
        """
        :param instance: A class already containing the obj list and the animal list, this takes precedence over the animal list and obj_list if provided.
        :param animal_list: A class containing the animal list
        :param obj_list: A class containing the object list
        :param node_of_interest: A list of strings of names of nodes to check for collisions, not rays but the key points.
        """


        if instance is not None:
            self._animal_list = instance.animal_list
            self._obj_list = instance.obj_list
        else:
            if animal_list is None or obj_list is None:
                raise ValueError("Must provide animal_list or obj_list, or an InstanceLoader")
            else:
                self._animal_list = animal_list
                self._obj_list = obj_list

        if not [type(*self._obj_list)].__contains__(trimesh.Trimesh):
            instance_list = [*self._animal_list.animals, *self._obj_list]
        else:
            instance_list = [*self._animal_list.animals]

        min_frame = min([inst.get_frame_range()[0] for inst in instance_list])
        max_frame = max([inst.get_frame_range()[1] for inst in instance_list])

        self._all_frames = range(min_frame, max_frame)

        self._node_of_interest = node_of_interest


        self._dt = np.dtype([('Frame', np.int32), ('Track', np.str_, 16), ('ID', np.uint8), ('Limb', np.str_, 16), ('Norm', np.float64, 3),
                             ('Point', np.float64, 3)])

        collision_list = self._calculate_collision_mt()
        df = DataFrame(collision_list.flatten().tolist(), columns=self._dt.names)
        df.set_index(["Frame", "Track"], inplace=True)
        self._collision_df = df


    def visualise_collision_rays(self, frame_idx: int) -> (np.ndarray, np.ndarray):
        """ Give a frame index, calculate collision points then visualise"""

        try:
            collision_sample = self._collision_df.loc[frame_idx]
        except KeyError as e:
            logger.info(f"CollisionDetector: KeyError: {e}")
            return None
        else:
            if len(collision_sample) == 0: #This shouldn't be reached but is error catching.
                logger.info("No collisions found for %d" % frame_idx)
                return None
            else:
                ray_array = np.empty((len(collision_sample),2, 3), dtype=np.float64)
                animal_name = []
                for i, contact in enumerate(collision_sample.itertuples()):
                    animal_name.append(contact.Index)
                    ray_array[i] = [contact.Point.astype(np.float64), contact.Point.astype(np.float64) + contact.Norm.astype(np.float64)]
                ray_visualise = trimesh.load_path(ray_array)

        return ray_visualise, animal_name

    def get_track(self, track_name: str) -> DataFrame:
        """ Return the collision Dataframe for only 1 animal
         Example usage:
         >>> cd = CollisionDetector()
         >>> cd.get_track('track99')
         """
        return self._collision_df.xs(track_name, level=1)

    def get_link(self, link_name_list: [str]) -> DataFrame:
        """ Return the collision Dataframe for only 1 animal
         Example usage:

         >>> cd = CollisionDetector()
         >>> cd.get_link(['neck_to_m_R0','neck_to_eye_R'])

         """
        return self._collision_df.loc[self._collision_df["Limb"].isin(link_name_list)]


    def _calculate_collisions(self, frame_idx: int):
        '''
        Given a frame index, calculate and return the collisions with the object and the surface normal at the
        point of collision.
        '''

        animals_in_frame = self._animal_list.where_frame_exist(frame_idx)
        _collision_array = np.empty(100, dtype=self._dt)
        n_collisions = 0
        for obj in self._obj_list: #TODO: currently only one object at a time
            if type(obj) is trimesh.Trimesh:
                geom = obj
            else:
                # Check that the object is in the frame
                if not obj.check_frame_exist(frame_idx):
                    continue
                geom = obj.generate_scene(frame_idx).to_mesh()
            # ri = trimesh.ray.ray_pyembree.RayMeshIntersector(geom)
            ri = trimesh.ray.ray_triangle.RayMeshIntersector(geom)

            for animal in animals_in_frame:
                #Check that the animal is in the frame

                pose_ray_dict = animal.get_pose_ray(frame_idx)

                if len(pose_ray_dict) > 0:

                    link_list = [*pose_ray_dict.keys()]

                    index_tri, index_ray, location = ri.intersects_id(
                        ray_origins=[pose_ray_dict[link]['origin'] for link in link_list],
                        ray_directions=[pose_ray_dict[link]['vector'] for link in link_list],
                        multiple_hits=False,
                        max_hits=10,
                        return_locations=True
                    )

                    if len(index_ray) > 0:

                        animal_collision = np.empty(len(index_ray), dtype=self._dt)
                        animal_collision['Frame'] = frame_idx
                        animal_collision['Track'] = animal.name
                        animal_collision['ID'] = np.array([*range(len(index_ray))])
                        animal_collision['Limb'] = [link_list[ray] for ray in index_ray]
                        animal_collision['Norm'] = geom.face_normals[index_tri].squeeze().astype(np.float64)
                        animal_collision['Point'] = location.astype(np.float64)

                        # collision_array = np.concatenate((collision_array, animal_collision), axis=0)
                        _collision_array[n_collisions: n_collisions + len(index_ray)] = animal_collision
                        n_collisions += len(index_ray)

        return _collision_array[np.argwhere(_collision_array)].flatten()


    def _calculate_collision_mt(self):
        """ Given an animal and object, h"""
        # scan all the file names in this directory

        frame_list = self._all_frames
        _collision_array = np.empty(10000, dtype=self._dt) #Preallocate an empty array up to 10000 collisions
        n_collisions = 0
        with (concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor):
            future_to_path = {executor.submit(self._calculate_collisions, frame_idx): frame_idx for frame_idx in frame_list}
            for future in concurrent.futures.as_completed(future_to_path):
                frame_input = future_to_path[future]
                try:
                    frame_collision_array = future.result()
                except Exception as exc:
                    logger.exception('%r generated an exception:' % (frame_input), exc_info=exc)
                else:
                    if not (frame_collision_array is None or len(frame_collision_array) == 0) :
                        _collision_array[n_collisions: n_collisions+len(frame_collision_array)] = frame_collision_array
                        # = np.concatenate((_collision_array, frame_collision_array), axis=0)
                        n_collisions += len(frame_collision_array)

        return _collision_array[np.argwhere(_collision_array)]

    def _calculate_collision_st(self):
        """ Given a path to a folder containing ".obj" or ".dae" files with the name of the corresponding frame, load these and
        convert to a dict of trimesh"""
        # scan all the file names in this directory

        frame_list = self._all_frames
        _collision_array = np.empty(10000, dtype=self._dt) #Preallocate an empty array up to 10000 collisions
        n_collisions = 0

        for frame_idx in frame_list:
            try:
                frame_collision_array = self._calculate_collisions(frame_idx)
            except Exception as exc:
                logger.exception('%r generated an exception:' % (frame_idx), exc_info=exc)
            else:
                if not (frame_collision_array is None or len(frame_collision_array) == 0) :
                    _collision_array[n_collisions: n_collisions+len(frame_collision_array)] = frame_collision_array
                    # = np.concatenate((_collision_array, frame_collision_array), axis=0)
                    n_collisions += len(frame_collision_array)

        return _collision_array[np.argwhere(_collision_array)]