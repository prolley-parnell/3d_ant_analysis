import logging
from typing import Optional
from pathlib import Path

import numpy as np
import trimesh

from src.animal import AnimalStruct, AnimalList
from src.object import CollisionObj, CollisionObjTransform
from src.loader import InstanceLoader

from pandas import DataFrame, concat
from pandas import MultiIndex
from pandas import Series

import concurrent.futures


logger = logging.getLogger(__name__)


class CollisionDetector:

    def __init__(self, instance: Optional[InstanceLoader] = None,
                 animal_list: Optional[AnimalList] = None,
                 obj_list: Optional[list[CollisionObj | CollisionObjTransform]] = None):

        if instance is not None:
            self._animal_list = instance.animal_list
            self._obj_list = instance.obj_list
        else:
            if animal_list or obj_list is None:
                raise ValueError("Must provide animal_list or obj_list, or an InstanceLoader")
            else:
                self._animal_list = animal_list
                self._obj_list = obj_list

        min_frame = min([inst.get_frame_range()[0] for inst in [*self._animal_list.animals, *self._obj_list]])
        max_frame = max([inst.get_frame_range()[1] for inst in [*self._animal_list.animals, *self._obj_list]])

        self._all_frames = range(min_frame, max_frame)

        all_animal_name = self._animal_list.animal_name_list

        self._dt = np.dtype([('Frame', np.int32), ('Track', np.str_, 16), ('ID', np.uint8), ('Limb', np.str_, 16), ('Norm', np.float64, 3),
                             ('Point', np.float64, 3)])
        # self._mi = MultiIndex.from_product([all_animal_name, [], ["Limb", "Norm", "Point"]], names=['Track', 'Instance', 'Contact'])
        # self._collision_df = DataFrame(None, index=self._mi, columns=self._all_frames)

        collision_list = self._calculate_collision_mt()
        df = DataFrame(collision_list.tolist(), columns=self._dt.names)
        df.set_index(["Frame", "Track"], inplace=True)
        self._collision_df = df


    def visualise_collision(self, frame_idx: int):
        ''' Give a frame index, calculate collision points then visualise'''

        location_array, normal_array = self._calculate_collisions(frame_idx)
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

        for obj in self._obj_list: #TODO: currently only one object at a time
            # Check that the object is in the frame
            if obj.check_frame_exist(frame_idx):
                geom = obj.generate_scene(frame_idx).to_mesh()

                animals_in_frame = self._animal_list.where_frame_exist(frame_idx)
                collision_array = np.empty(0, dtype=self._dt)
                for animal in animals_in_frame:
                    #Check that the animal is in the frame

                    pose_ray_dict = animal.get_pose_ray(frame_idx)
                    link_list = [*pose_ray_dict.keys()]
                    if len(pose_ray_dict) > 0:
                        # ri = trimesh.ray.ray_pyembree.RayMeshIntersector(self._obj_dict[frame_idx])
                        ri = trimesh.ray.ray_triangle.RayMeshIntersector(geom)


                        index_tri, index_ray, location = ri.intersects_id(
                            ray_origins=[pose_ray_dict[link]['origin'] for link in link_list],
                            ray_directions=[pose_ray_dict[link]['vector'] for link in link_list],
                            multiple_hits=True,
                            max_hits=10,
                            return_locations=True
                        )

                        if len(index_ray) > 0:

                            animal_collision = np.empty(len(index_ray), dtype=self._dt)
                            animal_collision['Frame'] = frame_idx
                            animal_collision['Track'] = animal.name
                            animal_collision['ID'] = np.array([*range(len(index_ray))])
                            animal_collision['Limb'] = [link_list[ray] for ray in index_ray]
                            animal_collision['Norm'] = geom.face_normals[index_tri].squeeze()
                            animal_collision['Point'] = location


                            collision_array = np.concatenate((collision_array, animal_collision), axis=0)

                    return collision_array


    def _calculate_collision_mt(self):
        """ Given a path to a folder containing ".obj" or ".dae" files with the name of the corresponding frame, load these and
        convert to a dict of trimesh"""
        # scan all the file names in this directory

        frame_list = self._all_frames
        _collision_array = np.empty(0, dtype=self._dt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_path = {executor.submit(self._calculate_collisions, frame_idx): frame_idx for frame_idx in frame_list}
            for future in concurrent.futures.as_completed(future_to_path):
                frame_input = future_to_path[future]
                try:
                    frame_collision_array = future.result()
                except Exception as exc:
                    logger.error('%r generated an exception: %s' % (frame_input, exc))
                else:
                    if not (frame_collision_array is None or len(frame_collision_array) == 0) :
                        _collision_array = np.concatenate((_collision_array, frame_collision_array), axis=0)

        return _collision_array