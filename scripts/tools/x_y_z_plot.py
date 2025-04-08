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
            self.ax_x.plot('x', data=data_frame.T[kp], label=kp)
            self.ax_y.plot('y', data=data_frame.T[kp], label=kp)
            self.ax_z.plot('z', data=data_frame.T[kp], label=kp)
        self.ax_x.legend()
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
            self.ax.plot(kp, data=data_frame.T, label=kp)
        self.ax.legend()
        self.ax.set_xlabel('Frame')

class KPVelocityMag(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 5):
        """
        Plot class to visualise the velocity of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 5 steps, not necessarily 5 frames difference.
        """

        velocity_df, kp_in_df = get_velocity_mag_df(animal, frames, node, window_size)
        super().__init__(velocity_df, kp_in_df)
        self.ax.set_ylabel('Velocity Magnitude')
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

        velocity_df, kp_in_df = get_velocity_xyz_df(animal, frames, node, window_size)
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
    shift_keys = position_df.keys()[window_size:]

    velocity_df = position_df[shift_keys].copy()

    for key_lower, key_upper in zip(keys, shift_keys):
        dt = key_upper - key_lower
        velocity_df[key_upper] = (position_df.loc[kp_in_df,key_upper] - position_df.loc[kp_in_df,key_lower]) / dt

    return velocity_df, kp_in_df

def get_velocity_mag_df(animal: AnimalStruct, frames: list[int] = None, node_list: list[str] = None, window_size: int = 3) -> (pd.DataFrame, list[str]):
    """
    Return the velocity for the provided keypoints across the total frames as the magnitude of the velocity abstracted from axis
    :param animal: An instance of AnimalStruct
    :param frames: A list of frame numbers to plot, defaults to all frames
    :param node_list: A list of node names to plot, defaults to None, resulting in all nodes plotted
    :param window_size: The size of the window size to use to get the average velocity, defaults to 3
    :return:
    """

    position_df, kp_in_df = animal.get_xyz_df(frames, node_list)


    keys = position_df.keys()[:-window_size]
    shift_keys = position_df.keys()[window_size:]

    velocity_df = pd.DataFrame(None, index=kp_in_df, columns=shift_keys)

    for key_lower, key_upper in zip(keys, shift_keys):
        dt = key_upper - key_lower
        for kp in kp_in_df:
            velocity_df.loc[kp,key_upper] = np.linalg.norm(position_df.loc[kp,key_upper] - position_df.loc[kp,key_lower]) / dt

    return velocity_df, kp_in_df





