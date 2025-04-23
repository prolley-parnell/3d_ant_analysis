### Class of a plot to show the X, Y, Z against the frame number for velocity or position
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import matplotlib.pyplot as plt
import pandas as pd
from src.animal import AnimalStruct

class XYZPlot:
    def __init__(self, data_frame, kp_list):
        """
        Plot tool class, pass  DataFrame object with the top level labels as the name of the nodes in kp_list, and the second
        level labels as 'x' and 'y' and 'z'. Plot all columns included
        :param data_frame: a DataFrame object containing position or velocity data.
        :param kp_list: the list of keypoints to plot.
        """

        self.fig, (self.ax_x, self.ax_y, self.ax_z) = plt.subplots(3,1,figsize=(9, 9), sharex=True)

        for kp in kp_list:
            kp_df = data_frame.T[kp].dropna()
            self.ax_x.plot('x', data=kp_df, label=kp)
            self.ax_y.plot('y', data=kp_df, label=kp)
            self.ax_z.plot('z', data=kp_df, label=kp)
        self.ax_x.legend()
        # max_y = np.nanmax(data_frame.values)
        # min_y = np.nanmin(data_frame.values)
        # self.ax_x.set_ylim(min_y, max_y)
        # self.ax_y.set_ylim(min_y, max_y)
        # self.ax_z.set_ylim(min_y, max_y)
        self.ax_z.set_xlabel('Frame')


class MagnitudePlot:
    def __init__(self, data_frame, kp_list):
        """
        Plot tool class, pass  DataFrame object with the top level labels as the name of the nodes in kp_list.
         Plot all columns included
        :param data_frame: a DataFrame object containing velocity data.
        :param kp_list: the list of keypoints to plot.
        """

        self.fig, self.ax = plt.subplots(1,1,figsize=(9, 3), sharex=True)

        for kp in kp_list:
            self.ax.plot(kp, data=data_frame.T.dropna(), label=kp)
        self.ax.legend()
        self.ax.set_xlabel('Frame')


class KPDisplaceMag(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 5):
        """
        Plot class to visualise the displacement of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 5 steps, not necessarily 5 frames difference.
        """

        position_df, kp_in_df = animal.get_xyz_df(frames, node)
        magnitude_df = xyz_to_mag_df(position_df, kp_in_df)
        super().__init__(magnitude_df, kp_in_df)
        self.ax.set_ylabel('Displacement Magnitude')
        plt.show()

class KPVelocityMag(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 5):
        """
        Plot class to visualise the velocity of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 5 steps, not necessarily 5 frames difference.
        """

        velocity_df, acceleration_df, kp_in_df = get_velocity_xyz_df(animal, frames, node, window_size)
        magnitude_df = xyz_to_mag_df(velocity_df, kp_in_df)
        super().__init__(magnitude_df, kp_in_df)
        self.ax.set_ylabel('Velocity Magnitude')
        plt.show()

class KPAccelerationMag(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 5):
        """
        Plot class to visualise the acceleration of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 5 steps, not necessarily 5 frames difference.
        """

        velocity_df, acceleration_df, kp_in_df = get_velocity_xyz_df(animal, frames, node, window_size)
        magnitude_df = xyz_to_mag_df(acceleration_df, kp_in_df)
        super().__init__(magnitude_df, kp_in_df)
        self.ax.set_ylabel('Acceleration Magnitude')
        plt.show()


class KPAccXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 5):
        """
        Plot class to visualise the acceleration of animal keypoints across the frames in the x, y, and z axes.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 5 steps, not necessarily 5 frames difference.
        """

        velocity_df, acceleration_df, kp_in_df = get_velocity_xyz_df(animal, frames, node, window_size)
        super().__init__(acceleration_df, kp_in_df)
        self.ax_x.set_ylabel('X Acceleration')
        self.ax_y.set_ylabel('Y Acceleration')
        self.ax_z.set_ylabel('Z Acceleration')
        plt.show()

class KPVelocityXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 5):
        """
        Plot class to visualise the velocity of animal keypoints across the frames in the x, y, and z axes.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 5 steps, not necessarily 5 frames difference.
        """

        velocity_df, acceleration_df, kp_in_df = get_velocity_xyz_df(animal, frames, node, window_size)
        super().__init__(velocity_df, kp_in_df)
        self.ax_x.set_ylabel('X Velocity')
        self.ax_y.set_ylabel('Y Velocity')
        self.ax_z.set_ylabel('Z Velocity')
        plt.show()

class KPPosition(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None):
        """
        Plot class instance for visualising the position in X, Y, and Z coordinates.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        """

        position_df, kp_in_df = animal.get_xyz_df(frames, node)
        super().__init__(position_df, kp_in_df)
        self.ax_x.set_ylabel('X Position')
        self.ax_y.set_ylabel('Y Position')
        self.ax_z.set_ylabel('Z Position')
        plt.show()


def get_velocity_xyz_df(animal: AnimalStruct, frames: list[int] = None, node_list: list[str] = None, window_size: int = 3) -> (pd.DataFrame, list[str]):
    """
    Return the velocity for the provided keypoints across the total frames in x,y and z
    :param animal: An instance of AnimalStruct
    :param frames: A list of frame numbers to plot, defaults to all frames
    :param node_list: A list of node names to plot, defaults to None, resulting in all nodes plotted
    :param window_size: The size of the window size to use to get the average velocity, defaults to 3
    :return: (pd.DataFrame, list[str]) velocity dataframe, list of node names
    """

    position_df, kp_in_df = animal.get_xyz_df(frames, node_list)

    keys = position_df.keys()[:-window_size]

    velocity_df = pd.DataFrame().reindex_like(position_df[keys])
    acceleration_df = pd.DataFrame(dtype=np.float64).reindex_like(position_df[keys])

    u = 0
    for key in keys:
        key_upper = key + window_size
        for kp in kp_in_df:
            p_start = position_df.loc[kp, key]
            if not p_start.isnull().any():
                if position_df.keys().__contains__(key_upper):
                    p_end = position_df.loc[kp, key_upper]
                    if not p_end.isnull().any():
                        s = p_end - p_start
                        dt = key_upper - key
                        a = np.asarray((2 * (s - u * dt)) / (dt ** 2), dtype=np.float64)

                        acceleration_df.loc[kp, key] = a
                        acceleration_df.infer_objects(copy=False).ffill(axis=1, limit=window_size, inplace=True)
                        v = u + a * dt
                        velocity_df.loc[kp, key] = v
                        velocity_df.infer_objects(copy=False).ffill(axis=1, limit=window_size, inplace=True)
                        u = v
                    else:
                        print(f"End point is invalid: Node: {kp}, Frame: {key_upper}")
            else:
                print(f"Starting point is invalid: Node: {kp}, Frame: {key}")


    return velocity_df, acceleration_df, kp_in_df

def xyz_to_mag_df(dataframe_in: pd.DataFrame, node_list: list[str] = None) -> pd.DataFrame:
    """ Convert a dataframe generated in the xyz into a magnitude dataframe """

    dataframe_out = pd.DataFrame().reindex_like(dataframe_in)
    keys = dataframe_in.keys()
    for key in keys:
        for kp in node_list:
            xyz = dataframe_in.loc[kp, key]
            if not xyz.isnull().any():
                dataframe_out.loc[kp,key] = np.linalg.norm(xyz)

    return dataframe_out
