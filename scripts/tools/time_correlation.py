## A class with functions to calculate and plot the time correlation for different nodes

import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=150)
import matplotlib.pyplot as plt
import pandas as pd
from src.animal import AnimalStruct
from scripts.tools.animal_dataframe import AnimalDataFrame
import logging
logger = logging.getLogger(__name__)


class MagnitudePlot:
    def __init__(self, data_frame, node_pair_list):
        """
        Plot tool class, pass  DataFrame object with the top level labels as the name of the nodes in kp_list.
         Plot all columns included
        :param data_frame: a DataFrame object containing velocity data.
        :param kp_list: the list of keypoints to plot.
        """

        n_ax = len(node_pair_list)
        self.fig, self.ax = plt.subplots(n_ax,1,figsize=(7, 3*n_ax),
                                         layout='constrained',
                                         sharex=False,
                                         sharey=True)

        for i, node_pair in enumerate(node_pair_list):
            self.ax[i].plot("%s_%s" % node_pair, data=data_frame.T.dropna(), label=['x', 'y', 'z'])
            self.ax[i].set_xlabel(f'{node_pair[1]} with respect to {node_pair[0]}')
        self.ax[0].legend()




class CorrelationPlot(MagnitudePlot):
    def __init__(self, animal: AnimalStruct, node_pairs: list[tuple[str, str]] = [('a_L2', 'm_L1')], min_max: tuple[int, int] = (-50,50), frame_range: tuple[int, int] = None, frame_indices: list[int]=None):

        if frame_range is None:
            frame_range = animal.frames

        frame_full = [*range(np.min(frame_range), np.max(frame_range) + 1)]

        if frame_indices is None:
            frame_indices = frame_full

        frame_indices = np.unique(frame_indices)

        adf = AnimalDataFrame(animal, frame_full, np.unique(node_pairs).tolist())
        position_df = adf.position_xyz(clean=True)

        #TODO Replace Position DF Nodes

        index_position_df = pd.DataFrame(None, columns=position_df.columns, index=position_df.index, dtype=np.float64)
        index_position_df.loc[:, frame_indices] = position_df.loc[:, frame_indices].values
        step = range(*min_max)
        mi = pd.MultiIndex.from_product([["%s_%s" % node_pair for node_pair in node_pairs], ['x', 'y', 'z']],
                                        names=['Node', 'Axis'])
        correlation_df = pd.DataFrame(None, index=mi, columns=step, dtype=np.float64)

        for shift in step:
            shifted_df = index_position_df.shift(periods=shift, axis='columns')
            for (node_a, node_b) in node_pairs:
                correlation_df.loc[["%s_%s" % (node_a, node_b)], shift] = position_df.loc[node_a].corrwith(
                    shifted_df.loc[node_b], axis=1, drop=False, method='spearman').values

        #There is an issue with the fragmentation of the dataframe here as I shift the dataframe and create N empty frames. This might be resolved by reindexing instead of shifting

        self.correlation_df = correlation_df

        super().__init__(correlation_df, node_pairs)

        mid_x = (self.fig.subplotpars.right + self.fig.subplotpars.left) / 2
        mid_y = (self.fig.subplotpars.top + self.fig.subplotpars.bottom) / 2
        self.fig.supylabel('Pearsons Correlation', y=mid_y)
        self.fig.supxlabel('Frame Offset', x=mid_x)
        plt.show()

