# From all the joint positions across time, find the bounding box of the tips of the antennae and the feet
#Find the bounding box of all points, and remove outliers then use the centre of the box as an approximation for the ground plane

from src.animal import AnimalList
import numpy as np
import open3d as o3d
import trimesh.creation

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

        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))

        new_p, le = p.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.2)

        self._pcl = new_p

        bb = new_p.get_oriented_bounding_box()
        bb.color = (0, 1, 0)

        self._bb = bb

        tform = np.eye(4)
        tform[:3, :3] = bb.R
        tform[:3, 3] = bb.center

        self._tform = tform

    def visualise_bounding_box(self):
        center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=self._tform[:3,3])
        o3d.visualization.draw_geometries([self._pcl, self._bb, center_frame])

    @property
    def tform(self):
        return self._tform

    def get_ground_collision(self):
        """ Return a trimesh cube representing the ground plane."""
        # Use the bounding box, but half extents and transformed to match the centre
        modified_tf = self._tform
        modified_tf[3, 3] += self._bb.extent[2] / 2
        box = trimesh.creation.box(extents=self._bb.extent, transform=modified_tf)
        return box

