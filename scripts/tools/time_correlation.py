## A class with functions to calculate and plot the time correlation for different nodes

import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from src.animal import AnimalStruct


class MagnitudePlot:
    def __init__(self, data_frame, kp_list):
        """
        Plot tool class, pass  DataFrame object with the top level labels as the name of the nodes in kp_list.
         Plot all columns included
        :param data_frame: a DataFrame object containing velocity data.
        :param kp_list: the list of keypoints to plot.
        """

        self.fig, self.ax = plt.subplots(1,1,figsize=(9, 3), sharex=True)

        for axis in data_frame.index.values:
            self.ax.plot(axis, data=data_frame.T.dropna(), label=axis)
        self.ax.legend()
        self.ax.set_xlabel('Frame Offset')


class CorrelationPlot(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, node_a: str = 'a_L2', node_b: str = 'm_L1', min_max: tuple[int, int] = (-50,50), frame_range: tuple[int, int] = None):

        if frame_range is None:
            frame_range = animal.frames

        frame_full = [*range(np.min(frame_range), np.max(frame_range) + 1)]

        position_df, kp_in_df = animal.get_xyz_df(frame_full, [node_a, node_b])

        step = range(*min_max)
        correlation_df = pd.DataFrame(None, index=['x', 'y', 'z'], columns=step)



        for shift in step:
            correlation_df.loc[:, shift] = position_df.loc[node_a].corrwith(
                position_df.loc[node_b].shift(periods=shift, axis='columns'), axis=1, drop=False)

        #There is an issue with the fragmentation of the dataframe here as I shift the dataframe and create N empty frames. This might be resolved by reindexing instead of shifting

        self.correlation_df = correlation_df

        super().__init__(correlation_df, kp_in_df)

        self.ax.set_ylabel('Pearsons Correlation')
        plt.show()




# Take one df, find correlation for time stem +N (n may be negative), find the median correlation for xyz separately per node

# Then plot the xyz correlation against N