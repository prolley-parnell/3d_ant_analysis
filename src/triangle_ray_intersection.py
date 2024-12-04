import logging
import numpy as np
import concurrent.futures
import trimesh
import os

logger = logging.getLogger(__name__)


class CollisionDetector:

    def __init__(self, obj_folder: str, skeleton_json_path: str, pose_csv: str):
        self._obj_folder = obj_folder
        self._obj_dict = {}
        self._read_obj_folder_mt(self._obj_folder)
        self._skeleton_connectivity = [(0, 1), (0, 17), (0, 16), (0, 19), (0, 18), (1, 14), (1, 15), (1, 22), (1, 23), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 4), (15, 2), (16, 8), (17, 6), (18, 12), (19, 10), (20, 1), (21, 1), (22, 24), (23, 25)]
        self._pose_array = np.loadtxt(pose_csv, delimiter=',')
        self._pose_dict = self._pose_csv_to_dict(self._pose_array)
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
                    print('%r generated an exception: %s' % (path_input, exc))
                else:
                    print('Frame index is %d ' % frame_idx)
                    self._obj_dict[frame_idx] = trimesh_obj

    @staticmethod
    def _pose_csv_to_dict(pose_array):
        ''' Assumes the structure is _x,_y, _z, _score'''
        csv_col = ['frame', 'thorax_x', 'thorax_y', 'thorax_z', 'thorax_score', 'neck_x', 'neck_y', 'neck_z', 'neck_score', 'a_L1_x', 'a_L1_y', 'a_L1_z', 'a_L1_score', 'a_L2_x', 'a_L2_y', 'a_L2_z', 'a_L2_score', 'a_R1_x', 'a_R1_y', 'a_R1_z', 'a_R1_score', 'a_R2_x', 'a_R2_y', 'a_R2_z', 'a_R2_score', 'leg_f_L1_x', 'leg_f_L1_y', 'leg_f_L1_z', 'leg_f_L1_score', 'leg_f_L2_x', 'leg_f_L2_y', 'leg_f_L2_z', 'leg_f_L2_score', 'leg_f_R1_x', 'leg_f_R1_y', 'leg_f_R1_z', 'leg_f_R1_score', 'leg_f_R2_x', 'leg_f_R2_y', 'leg_f_R2_z', 'leg_f_R2_score', 'leg_m_L1_x', 'leg_m_L1_y', 'leg_m_L1_z', 'leg_m_L1_score', 'leg_m_L2_x', 'leg_m_L2_y', 'leg_m_L2_z', 'leg_m_L2_score', 'leg_m_R1_x', 'leg_m_R1_y', 'leg_m_R1_z', 'leg_m_R1_score', 'leg_m_R2_x', 'leg_m_R2_y', 'leg_m_R2_z', 'leg_m_R2_score', 'a_R0_x', 'a_R0_y', 'a_R0_z', 'a_R0_score', 'a_L0_x', 'a_L0_y', 'a_L0_z', 'a_L0_score', 'leg_f_R0_x', 'leg_f_R0_y', 'leg_f_R0_z', 'leg_f_R0_score', 'leg_f_L0_x', 'leg_f_L0_y', 'leg_f_L0_z', 'leg_f_L0_score', 'leg_m_R0_x', 'leg_m_R0_y', 'leg_m_R0_z', 'leg_m_R0_score', 'leg_m_L0_x', 'leg_m_L0_y', 'leg_m_L0_z', 'leg_m_L0_score', 'eye_L_x', 'eye_L_y', 'eye_L_z', 'eye_L_score', 'eye_R_x', 'eye_R_y', 'eye_R_z', 'eye_R_score', 'm_L0_x', 'm_L0_y', 'm_L0_z', 'm_L0_score', 'm_R0_x', 'm_R0_y', 'm_R0_z', 'm_R0_score', 'm_L1_x', 'm_L1_y', 'm_L1_z', 'm_L1_score', 'm_R1_x', 'm_R1_y', 'm_R1_z', 'm_R1_score']
        link_names = ['thorax', 'neck', 'a_L1', 'a_L2', 'a_R1', 'a_R2', 'leg_f_L1', 'leg_f_L2', 'leg_f_R1', 'leg_f_R2', 'leg_m_L1', 'leg_m_L2', 'leg_m_R1', 'leg_m_R2', 'a_R0', 'a_L0', 'leg_f_R0', 'leg_f_L0', 'leg_m_R0', 'leg_m_L0', 'eye_L', 'eye_R', 'm_L0', 'm_R0', 'm_L1', 'm_R1']
        pose_dict = {}
        for row in pose_array:
            pose_dict[row[0]] = {}
            for name in link_names:
                pose_dict[row[0]][name] = {}
                x_mask = [ name+"_x" in csv_name for csv_name in csv_col]
                y_mask = [ name+"_y" in csv_name for csv_name in csv_col]
                z_mask = [ name+"_z" in csv_name for csv_name in csv_col]
                pose_dict[row[0]][name]['xyz'] = np.array([row[x_mask][0],row[y_mask][0],row[z_mask][0]])
                score_mask = [name + "_score" in csv_name for csv_name in csv_col]
                pose_dict[row[0]][name]['score'] = row[score_mask][0]

        return pose_dict

    def _sleap_json_to_dict(self, skeleton_file: str):
        ''' Given path to JSON file describing skeleton, convert to dict with connectivity'''



    @staticmethod
    def _single_obj_to_trimesh(obj_path: str):
        ''' Convert the obj at the given path to a trimesh object and return the object and frame index '''
        if not obj_path.startswith('.') and os.path.splitext(obj_path)[-1] in [".obj", ".OBJ"]:
            frame_index = int(os.path.splitext(obj_path)[0])
            return frame_index, trimesh.load("example_obj_folder/"+obj_path, force='mesh')


    def get_collisions(self, frame_idx: int):
        '''
        Given a frame index, calculate and return the collisions with the object and the surface normal at the
        point of collision.
        '''
        # if not self._check_frame_exist(frame_idx):
        #     logger.error('Could not find collision for frame index {}'.format(frame_idx))
        #     return None

        if not self._pose_ray_dict.keys().__contains__(frame_idx):
            self._pose_ray_dict[frame_idx] = self._generate_rays(frame_idx)

        pose_ray_dict = self._pose_ray_dict[frame_idx]
        ri = trimesh.ray.ray_pyembree.RayMeshIntersector(self._obj_dict[244])

        index_tri, index_ray, location = ri.intersects_id(
            ray_origins=[pose_ray_dict[link]['origin'] for link in pose_ray_dict.keys()],
            ray_directions=[pose_ray_dict[link]['vector'] for link in pose_ray_dict.keys()],
            multiple_hits=True,
            max_hits=10,
            return_locations=True
        )



        return location, surface_norm

    def _get_surface_norm(self, frame_idx, triangle_idx):
        ''' Find the surface normal for the object at the frame index listed and for the triangle face'''



    def _check_frame_exist(self, frame_idx):
        ''' If frame index is not present in one of the dictionaries, return false'''
        if not self._obj_dict.keys().__contains__(frame_idx):
            logger.error('Frame index {} not present in the object mesh dictionary \n'.format(frame_idx))
            return False
        if not self._pose_dict.keys().__contains__(frame_idx):
            logger.error('Frame index {} not present in the animal skeleton dictionary \n'.format(frame_idx))
            return False
        return True


    def _generate_rays(self, frame_idx):
        ''' For the skeleton at the frame index provided, generate a series of origins and vector axes to represent the
         links between the key nodes'''
        ray_dict = {}

        # Find all links in skeleton and assign names
        for pair in self._skeleton_connectivity:

            name_a = list(self._pose_dict[frame_idx].keys())[pair[0]]
            name_b = list(self._pose_dict[frame_idx].keys())[pair[1]]

            # Assign the direction based on distance to core/anchor node
            point_a = self._pose_dict[frame_idx][name_a]['xyz']
            point_b = self._pose_dict[frame_idx][name_b]['xyz']

            if any(np.isnan(point_a)) or any(np.isnan(point_b)):
                logging.info("Point is NaN")
            else:
                #TODO No ordering yet

                # Generate the vector and origin based on this direction
                origin = point_a
                vector = point_b - point_a

                # Assign vector and origin to key name
                name = name_a + "_to_" + name_b
                ray_dict[name] = {}

                ray_dict[name]['origin'] = origin
                ray_dict[name]['vector'] = vector

        return ray_dict





#
# def find_point_on_obj_normal_id(fbTriang, samplePoint):
#     """
#     Find normal at a given contact point on a mesh object and return various information about the contact.
#
#     Parameters:
#     - fbTriang: trimesh.Trimesh object representing the triangulated mesh.
#     - samplePoint: nx3 numpy array of points on the mesh for which we need to find the normal and other details.
#
#     Returns:
#     - pointOnObj: n x 3 numpy array of the Cartesian coordinates of the point on the object.
#     - normalVArray: n x 3 numpy array of the surface normals at each contact point.
#     - vertIDArray: numpy array of vertex IDs where the collision occurred at a vertex.
#     - faceIDArray: numpy array of face IDs where the collision occurred on a face plane.
#     """
#
#     nPoint = samplePoint.shape[0]
#     pointOnObj = np.full_like(samplePoint, np.nan)
#     normalVArray = np.full_like(samplePoint, np.nan)
#     vertIDArray = np.full((nPoint,), np.nan)
#     faceIDArray = np.full((nPoint,), np.nan)
#
#     # Build a k-d tree for nearest neighbor search
#     kdtree = cKDTree(fbTriang.vertices)
#
#     for n in range(nPoint):
#         # Find nearest vertex to the sample point
#         dist, nearVertID = kdtree.query(samplePoint[n, :])
#         nearV_cartesian = fbTriang.vertices[nearVertID, :]
#         nearV_dist = np.linalg.norm(nearV_cartesian - samplePoint[n, :])
#
#         if round(nearV_dist, 8) == 0:
#             # The point of contact is on the vertex
#             normalVArray[n, :] = fbTriang.vertex_normals[nearVertID]
#             pointOnObj[n, :] = fbTriang.vertices[nearVertID, :]
#             vertIDArray[n] = nearVertID
#         else:
#             # The contact point is on a face, we will calculate barycentric coordinates
#             nearV_neighbours = fbTriang.vertex_neighbors[nearVertID]
#             repTestPt = np.tile(samplePoint[n, :], (len(nearV_neighbours), 1))
#
#             # Barycentric coordinates calculation (this part may require implementation or modification)
#             B = fbTriang.barycentric_coordinates(nearV_neighbours, repTestPt)
#
#             # Round the barycentric coordinates to avoid precision issues
#             roundB = np.round(B, 9)
#
#             # Convert back to Cartesian coordinates from barycentric
#             CB = fbTriang.to_cartesian(nearV_neighbours, roundB)
#             CBDist = np.linalg.norm(CB - samplePoint[n, :], axis=1)
#
#             zero_id = np.where(np.round(CBDist, 6) == 0)[0]
#
#             if len(zero_id) > 1:
#                 pointOnObj[n, :] = CB[zero_id[0], :]
#                 faceIDArray[n] = nearV_neighbours[zero_id[0]]
#                 multiNorm = fbTriang.face_normals[nearV_neighbours[zero_id]]
#                 normalVArray[n, :] = np.mean(multiNorm, axis=0)
#
#             elif len(zero_id) == 1:
#                 pointOnObj[n, :] = CB[zero_id[0], :]
#                 faceIDArray[n] = nearV_neighbours[zero_id[0]]
#                 normalVArray[n, :] = fbTriang.face_normals[faceIDArray[n]]
#
#             else:
#                 # Find the closest face if no exact match
#                 nearestID = np.argmin(CBDist)
#                 pointOnObj[n, :] = CB[nearestID, :]
#                 faceIDArray[n] = nearV_neighbours[nearestID]
#                 normalVArray[n, :] = fbTriang.face_normals[faceIDArray[n]]
#
#     return pointOnObj, normalVArray, vertIDArray, faceIDArray


# class TriangleRayIntersect:
#     @staticmethod
#     def _ensure_shape(arr):
#         if arr.ndim == 1:
#             return arr[np.newaxis, :]
#         return arr
#
#     @staticmethod
#     def _replicate_if_needed(arr, size):
#         if arr.shape[0] == 1 and size > 1:
#             return np.tile(arr, (size, 1))
#         return arr
#
#     def calculate_intersection(self, orig, dir, vert0, vert1, vert2, **kwargs):
#         """
#         Ray-Triangle Intersection based on Möller-Trumbore algorithm.
#         Based on:
#         *"Fast, minimum storage ray-triangle intersection". Tomas Möller and
#             Ben Trumbore. Journal of Graphics Tools, 2(1):21--28, 1997.
#           http://www.graphics.cornell.edu/pubs/1997/MT97.pdf
#         * http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/
#         * http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/raytri.c
#
#          Author:
#             Jarek Tuszynski (jaroslaw.w.tuszynski@leidos.com)
#             Modified to Python by Persie Rolley-Parnell with the assistance of ChatGPT
#
#          License: BSD license (http://en.wikipedia.org/wiki/BSD_licenses)
#
#         Args:
#             orig (ndarray): Nx3 array representing the origins of rays.
#             dir (ndarray): Nx3 array representing the directions of rays.
#             vert0 (ndarray): Nx3 array of the first vertices of the triangles.
#             vert1 (ndarray): Nx3 array of the second vertices of the triangles.
#             vert2 (ndarray): Nx3 array of the third vertices of the triangles.
#             kwargs: Optional keyword arguments:
#                 - eps: tolerance for numerical errors (default = 1e-5)
#                 - plane_type: 'one sided' or 'two sided' triangles (default = 'two sided')
#                 - line_type: 'ray', 'line' or 'segment' (default = 'ray')
#                 - border: 'normal', 'inclusive' or 'exclusive' (default = 'normal')
#                 - full_return: boolean to control full output (default = False)
#
#         Returns:
#             tuple: intersect (bool array), t (float array), u (float array), v (float array), xcoor (ndarray)
#         """
#
#         # Convert inputs to proper shapes if necessary
#         orig = self._ensure_shape(orig)
#         dir = self._ensure_shape(dir)
#         vert0 = self._ensure_shape(vert0)
#         vert1 = self._ensure_shape(vert1)
#         vert2 = self._ensure_shape(vert2)
#
#         # Number of rays and triangles
#         n_ray_tri_max = max(orig.shape[0], dir.shape[0], vert0.shape[0], vert1.shape[0], vert2.shape[0])
#
#         # Replicate single rays or triangles to match size
#         orig = self._replicate_if_needed(orig, n_ray_tri_max)
#         dir = self._replicate_if_needed(dir, n_ray_tri_max)
#         vert0 = self._replicate_if_needed(vert0, n_ray_tri_max)
#         vert1 = self._replicate_if_needed(vert1, n_ray_tri_max)
#         vert2 = self._replicate_if_needed(vert2, n_ray_tri_max)
#
#         # Default arguments
#         eps = kwargs.get('eps', 1e-5)
#         plane_type = kwargs.get('plane_type', 'two sided')
#         line_type = kwargs.get('line_type', 'ray')
#         border = kwargs.get('border', 'normal')
#         full_return = kwargs.get('full_return', False)
#
#         # Set border handling
#         if border == 'normal':
#             zero = 0.0
#         elif border == 'inclusive':
#             zero = eps
#         elif border == 'exclusive':
#             zero = -eps
#         else:
#             raise ValueError('Border parameter must be either "normal", "inclusive" or "exclusive"')
#
#         # Initialize outputs
#         intersect = np.zeros(n_ray_tri_max, dtype=bool)
#         t = np.inf * np.ones(n_ray_tri_max)
#         u = t.copy()
#         v = t.copy()
#         xcoor = np.full((n_ray_tri_max, 3), np.nan)
#
#         # Compute edges and vectors
#         edge1 = vert1 - vert0
#         edge2 = vert2 - vert0
#         tvec = orig - vert0
#         pvec = np.cross(dir, edge2, axis=1)
#         det = np.sum(edge1 * pvec, axis=1)
#
#         # Check for parallel rays
#         if plane_type == 'two sided':
#             angle_ok = np.abs(det) > eps
#         elif plane_type == 'one sided':
#             angle_ok = det > eps
#         else:
#             raise ValueError('PlaneType must be "one sided" or "two sided"')
#
#         # If no intersections, return
#         if not np.any(angle_ok):
#             return intersect, t, u, v, xcoor
#
#         # Compute barycentric coordinates
#         u = np.sum(tvec * pvec, axis=1) / det
#         if full_return:
#             qvec = np.cross(tvec, edge1, axis=1)
#             v = np.sum(dir * qvec, axis=1) / det
#             t = np.sum(edge2 * qvec, axis=1) / det
#             ok = angle_ok & (u >= -zero) & (v >= -zero) & (u + v <= 1.0 + zero)
#         else:
#             v = np.full_like(u, np.nan)
#             t = np.full_like(u, np.nan)
#             ok = angle_ok & (u >= -zero) & (u <= 1.0 + zero)
#
#             if np.any(ok):
#                 qvec = np.cross(tvec[ok], edge1[ok], axis=1)
#                 v[ok] = np.sum(dir[ok] * qvec, axis=1) / det[ok]
#                 t[ok] = np.sum(edge2[ok] * qvec, axis=1) / det[ok]
#
#             ok &= (v >= -zero) & (u + v <= 1.0 + zero)
#
#         # Determine intersection based on line_type
#         if line_type == 'line':
#             intersect = ok
#         elif line_type == 'ray':
#             intersect = ok & (t >= -zero)
#         elif line_type == 'segment':
#             intersect = ok & (t >= -zero) & (t <= 1.0 + zero)
#         else:
#             raise ValueError('line_type parameter must be "line", "ray" or "segment"')
#
#         # Calculate intersection coordinates
#         if np.any(intersect):
#             xcoor[intersect, :] = (
#                     vert0[intersect, :] +
#                     edge1[intersect, :] *
#                     u[intersect][:, np.newaxis] +
#                     edge2[intersect,:] *
#                     v[intersect][:,np.newaxis])
#
#         return intersect, t, u, v, xcoor
