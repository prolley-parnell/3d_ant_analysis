### Class of a plot to show the X, Y, Z against the frame number for velocity or position
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from src.animal import AnimalStruct, AnimalDataFrame
import logging
logger = logging.getLogger(__name__)
#TODO: Update all the class descriptions to match updated function
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
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 1, filter_for_outlier=False):
        """
        Plot class to visualise the displacement of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 1 steps, not necessarily 1 frames difference.
        """

        adf = AnimalDataFrame(animal, frames, node, window_size)
        magnitude_df = adf.displace_mag(filter_for_outlier)

        super().__init__(magnitude_df, adf.kp_in_df)

        self.ax.set_ylabel('Displacement Magnitude')
        plt.show()

class KPVelocityMag(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 1, filter_for_outlier=False):
        """
        Plot class to visualise the velocity of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 1 steps, not necessarily 1 frames difference.
        """

        # position_df, velocity_df, acceleration_df, kp_in_df = animal.get_motion_df(frames, node, window_size, filter_for_outlier)
        # magnitude_df = xyz_to_mag_df(velocity_df, kp_in_df)

        adf = AnimalDataFrame(animal, frames, node, window_size)
        magnitude_df = adf.velocity_mag(filter_for_outlier)

        super().__init__(magnitude_df, adf.kp_in_df)
        self.ax.set_ylabel('Velocity Magnitude')
        plt.show()

class KPAccelerationMag(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 1, filter_for_outlier=False):
        """
        Plot class to visualise the acceleration of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 1 steps, not necessarily 1 frames difference.
        """

        adf = AnimalDataFrame(animal, frames, node, window_size)
        acceleration_df = adf.acceleration_mag(filter_for_outlier)
        super().__init__(acceleration_df, adf.kp_in_df)
        self.ax.set_ylabel('Acceleration Magnitude')
        plt.show()


class KPAccXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 1, filter_for_outlier=False):
        """
        Plot class to visualise the acceleration of animal keypoints across the frames in the x, y, and z axes.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 1 steps, not necessarily 1 frames difference.
        """

        adf = AnimalDataFrame(animal, frames, node, window_size)
        acceleration_df = adf.acceleration_xyz(filter_for_outlier)

        super().__init__(acceleration_df, adf.kp_in_df)
        self.ax_x.set_ylabel('X Acceleration')
        self.ax_y.set_ylabel('Y Acceleration')
        self.ax_z.set_ylabel('Z Acceleration')
        plt.show()

class KPVelocityXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 1, filter_for_outlier=False):
        """
        Plot class to visualise the velocity of animal keypoints across the frames in the x, y, and z axes.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param window_size: Time window to average over, defaults to 1 steps, not necessarily 1 frames difference.
        """

        adf = AnimalDataFrame(animal, frames, node, window_size)
        velocity_df = adf.velocity_xyz(filter_for_outlier)

        super().__init__(velocity_df, adf.kp_in_df)
        self.ax_x.set_ylabel('X Velocity')
        self.ax_y.set_ylabel('Y Velocity')
        self.ax_z.set_ylabel('Z Velocity')
        plt.show()


class KPDisplaceXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 1, filter_for_outlier=False):
        """
        Plot class instance for visualising the position in X, Y, and Z coordinates.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        """

        adf = AnimalDataFrame(animal, frames, node, window_size)
        displace_df = adf.displace_xyz(filter_for_outlier)

        super().__init__(displace_df, adf.kp_in_df)
        self.ax_x.set_ylabel('X Position')
        self.ax_y.set_ylabel('Y Position')
        self.ax_z.set_ylabel('Z Position')
        plt.show()

class KPPositionXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, window_size:int = 1, filter_for_outlier=False):
        """
        Plot class instance for visualising the position in X, Y, and Z coordinates.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        """

        adf = AnimalDataFrame(animal, frames, node, window_size)
        position_df = adf.position_xyz(filter_for_outlier)

        super().__init__(position_df, adf.kp_in_df)
        self.ax_x.set_ylabel('X Position')
        self.ax_y.set_ylabel('Y Position')
        self.ax_z.set_ylabel('Z Position')
        plt.show()

