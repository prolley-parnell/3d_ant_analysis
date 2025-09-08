# From all the joint positions across time, find the bounding box of the tips of the antennae and the feet
#Find the bounding box of all points, and remove outliers then use the centre of the box as an approximation for the ground plane

from src.animal import AnimalList
import numpy as np
import open3d as o3d
import trimesh.creation
import pyransac3d as pyrsc

class GroundPlaneEstimation:
    def __init__(self,
                 animal_list: AnimalList,
                 node_list=None):

        if node_list is None:
            node_list = ['leg_m_L2', 'leg_m_R2']

        all_points_list = []
        for animal in animal_list.animals:
            position_df, kp_in_df = animal.get_xyz_df(None, node_list)
            all_points_list.append(
                np.vstack([np.array(position_df.loc[node].T.dropna(), dtype=np.float64) for node in node_list]))
        all_points = np.vstack(all_points_list)

        #Method 1 : Oriented bounding box
        # p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
        # new_p, le = p.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
        # bb = new_p.get_oriented_bounding_box()
        # bb.color = (0, 1, 0)
        # self._pcl = new_p
        # self._bb = bb
        # tform = np.eye(4)
        # tform[:3, :3] = self._bb.R
        # tform[:3, 3] = self._bb.center

        #Method 2: PyRANSAC
        plane = pyrsc.Plane()
        best_eq, best_inliers = plane.fit(all_points, thresh=0.25, minPoints=np.floor(len(all_points_list)*0.7), maxIteration=10000)

        centre = np.mean(all_points[best_inliers], axis=0) #+ (np.asarray(best_eq[:3])*0.5) #scalar to see how offset from the plane to set the centre
        tform = trimesh.geometry.align_vectors([0, 0, 1], np.asarray(best_eq[:3]))
        tform[:3,3] = centre
        self._pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
        self._extent = [10, 10, 0.15]

        self._tform = tform

    def visualise_bounding_box(self):
        center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=self._tform[:3,3])
        box = o3d.geometry.OrientedBoundingBox(center=self._tform[:3, 3], R=self._tform[:3, :3], extent=self._extent)
        o3d.visualization.draw_geometries([self._pcl, box, center_frame])

    @property
    def tform(self):
        return self._tform

    def get_ground_collision(self):
        """ Return a trimesh cube representing the ground plane."""
        # Use the bounding box, but half extents and transformed to match the centre
        box = trimesh.creation.box(extents=self._extent, transform=self._tform)
        return box



