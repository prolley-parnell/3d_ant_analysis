### Class of a plot to show the X, Y, Z  or magnitude against the frame number for position, displacement, velocity or acceleration
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import matplotlib.pyplot as plt
from src.animal import AnimalStruct
from pandas import DataFrame
from scripts.tools.animal_dataframe import AnimalDataFrame
import logging
logger = logging.getLogger(__name__)

class XYZPlot:
    def __init__(self, data_frame: DataFrame, kp_list: [str]):
        """
        Plot tool class, pass  DataFrame object with the top level labels as the name of the nodes in kp_list, and the second
        level labels as 'x' and 'y' and 'z'. Plot all columns included
        :param data_frame: a DataFrame object containing position or velocity data.
        :param kp_list: the list of node names to plot.
        """

        self.fig, (self.ax_x, self.ax_y, self.ax_z) = plt.subplots(3,1,figsize=(9, 9), sharex=True)

        for kp in kp_list:
            #kp_df = data_frame.T[kp].dropna()
            kp_df = data_frame.T[kp]
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
    def __init__(self, data_frame: DataFrame, kp_list: [str]):
        """
        Plot tool class for a DataFrame including the magnitude of a variable
        DataFrame object has the top level labels as the name of the nodes in kp_list.
        Plot all columns included
        :param data_frame: a DataFrame object containing velocity data.
        :param kp_list: the list of node names to plot.
        """

        self.fig, self.ax = plt.subplots(1,1,figsize=(9, 3), sharex=True)

        for kp in kp_list:
            self.ax.plot(kp, data=data_frame.T, label=kp)
            # self.ax.plot(kp, data=data_frame.T.dropna(), label=kp)
        self.ax.legend()
        self.ax.set_xlabel('Frame')


class KPDisplaceMag(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, filter_for_outlier=False):
        """
        Plot class to visualise the displacement of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param filter_for_outlier: Whether to plot only inlier node values or not. If True, uses filtered position values.
        """

        adf = AnimalDataFrame(animal, frames, node)
        magnitude_df = adf.displace_mag(filter_for_outlier)

        super().__init__(magnitude_df, adf.kp_in_df)

        self.ax.set_ylabel('Displacement Magnitude')
        plt.show()

class KPVelocityMag(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, filter_for_outlier=False):
        """
        Plot class to visualise the velocity of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param filter_for_outlier: Whether to plot only inlier node values or not. If True, uses filtered position values.
        """

        adf = AnimalDataFrame(animal, frames, node)
        magnitude_df = adf.velocity_mag(filter_for_outlier)

        super().__init__(magnitude_df, adf.kp_in_df)
        self.ax.set_ylabel('Velocity Magnitude')
        plt.show()

class KPAccelerationMag(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, filter_for_outlier=False):
        """
        Plot class to visualise the acceleration of animal keypoints across the frames as a magnitude
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param filter_for_outlier: Whether to plot only inlier node values or not. If True, uses filtered position values.
        """

        adf = AnimalDataFrame(animal, frames, node)
        acceleration_df = adf.acceleration_mag(filter_for_outlier)
        super().__init__(acceleration_df, adf.kp_in_df)
        self.ax.set_ylabel('Acceleration Magnitude')
        plt.show()


class KPAccelerationXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, filter_for_outlier=False):
        """
        Plot class to visualise the acceleration of animal keypoints across the frames in the x, y, and z axes.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param filter_for_outlier: Whether to plot only inlier node values or not. If True, uses filtered position values.
        """

        adf = AnimalDataFrame(animal, frames, node)
        acceleration_df = adf.acceleration_xyz(filter_for_outlier)

        super().__init__(acceleration_df, adf.kp_in_df)
        self.ax_x.set_ylabel('X Acceleration')
        self.ax_y.set_ylabel('Y Acceleration')
        self.ax_z.set_ylabel('Z Acceleration')
        plt.show()

class KPVelocityXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, filter_for_outlier=False):
        """
        Plot class to visualise the velocity of animal keypoints across the frames in the x, y, and z axes.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param filter_for_outlier: Whether to plot only inlier node values or not. If True, uses filtered position values.
        """

        adf = AnimalDataFrame(animal, frames, node)
        velocity_df = adf.velocity_xyz(filter_for_outlier)

        super().__init__(velocity_df, adf.kp_in_df)
        self.ax_x.set_ylabel('X Velocity')
        self.ax_y.set_ylabel('Y Velocity')
        self.ax_z.set_ylabel('Z Velocity')
        plt.show()


class KPDisplaceXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, filter_for_outlier=False):
        """
        Plot class instance for visualising the displacement in X, Y, and Z coordinates.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param filter_for_outlier: Whether to plot only inlier node values or not. If True, uses filtered position values.
        """

        adf = AnimalDataFrame(animal, frames, node)
        displace_df = adf.displace_xyz(filter_for_outlier)

        super().__init__(displace_df, adf.kp_in_df)
        self.ax_x.set_ylabel('X Position')
        self.ax_y.set_ylabel('Y Position')
        self.ax_z.set_ylabel('Z Position')
        plt.show()

class KPPositionXYZ(XYZPlot):
    def __init__(self, animal: AnimalStruct, frames: list[int] = None, node: list[str] = None, filter_for_outlier=False):
        """
        Plot class instance for visualising the position in X, Y, and Z coordinates.
        :param animal: AnimalStruct instance to plot
        :param frames: list of frames to plot, defaults to None so all are plotted
        :param node: list of nodes to plot, defaults to None so all are plotted
        :param filter_for_outlier: Whether to plot only inlier node values or not. If True, uses filtered position values.
        """

        adf = AnimalDataFrame(animal, frames, node)
        position_df = adf.position_xyz(filter_for_outlier)

        super().__init__(position_df, adf.kp_in_df)
        self.ax_x.set_ylabel('X Position')
        self.ax_y.set_ylabel('Y Position')
        self.ax_z.set_ylabel('Z Position')
        plt.show()

