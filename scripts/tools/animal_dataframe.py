# Class to process an AnimalStruct object and the DataFrame associated with it to be used in plotting.
import logging
from src.animal import AnimalStruct
import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)

class AnimalDataFrame:
    def __init__(self,
                 animal: AnimalStruct,
                 frames: list[int] = None,
                 node_list: list[str] = None):
        """
        :param animal: The AnimalStruct object to use
        :param frames: A sequential frame window to query
        :param node_list: A list of nodes contained in the animal to query
        """

        self._animal = animal
        self._frames = frames
        self._node_list = node_list

        self._position_df, self._displace_df, self._velocity_df, self._acceleration_df, self._kp_in_df = self._get_motion_df()
        self._inlier_mask = self._calculate_inlier()

    def _get_motion_df(self) -> (DataFrame, DataFrame, DataFrame, DataFrame, list[str]) :
        """
        From the position dataframe, calculate the displacement, velocity and acceleration and return all as dataframes
        :return: position_df, displace_df, velocity_df, acceleration_df, kp_in_df, list of node names
        """

        position_df, kp_in_df = self._animal.get_xyz_df(self._frames, self._node_list)
        keys = position_df.keys()

        displace_df = DataFrame(dtype=np.float64, columns=position_df[keys].columns, index=position_df[keys].index)
        velocity_df = DataFrame(dtype=np.float64).reindex_like(position_df[keys])
        acceleration_df = DataFrame(dtype=np.float64).reindex_like(position_df[keys])

        for kp in kp_in_df:
            # u = 0
            for key in keys:
                key_upper = key + 1

                p_start = position_df.loc[kp, key]
                if not p_start.isnull().any():
                    if position_df.keys().__contains__(key_upper):
                        p_end = position_df.loc[kp, key_upper]
                        if not p_end.isnull().any():
                            s = np.asarray(p_end - p_start, dtype=np.float64)
                            dt = key_upper - key
                            v = np.asarray(s / dt, dtype=np.float64)

                            if velocity_df.keys().__contains__(key - 1):
                                u = velocity_df.loc[kp, key - 1]
                                dt = 1
                            else:
                                u = np.nan
                            a = np.asarray((v - u) / dt, dtype=np.float64)

                            displace_df.at[kp, key] = s
                            acceleration_df.at[kp, key] = a
                            velocity_df.at[kp, key] = v

                        else:
                            logger.info(f"End point is invalid: Node: {kp}, Frame: {key_upper}")
                else:
                    logger.info(f"Starting point is invalid: Node: {kp}, Frame: {key}")

        return position_df, displace_df, velocity_df, acceleration_df, kp_in_df


    def _calculate_inlier(self) -> DataFrame :
        """ Use Modified Z-Score of the displacement and velocity across frames to remove outliers from the data"""
        def modified_z_score(series):
            """ Calculate a Modified Z-Score, using mean absolute deviation, for a panda Series """
            median_y = series.median()
            median_absolute_deviation_y = (np.abs(series - median_y)).median()
            modified_z_scores = (series - median_y) / median_absolute_deviation_y
            return modified_z_scores

        acc_mag_df = self._xyz_to_mag_df(self._acceleration_df, self._kp_in_df)
        acc_zsc_mag = acc_mag_df.apply(modified_z_score, axis='columns', result_type='broadcast')
        acc_zsc_xyz = self._acceleration_df.apply(modified_z_score, axis='columns', result_type='broadcast')

        disp_mag_df = self._xyz_to_mag_df(self._displace_df, self._kp_in_df)
        disp_zsc_mag = disp_mag_df.apply(modified_z_score, axis='columns', result_type='broadcast')
        disp_zsc_xyz = self._displace_df.apply(modified_z_score, axis='columns', result_type='broadcast')

        acc_mag_inlier_mask = acc_zsc_mag[np.abs(acc_zsc_mag) > 3].groupby(level=0).count() == 0
        acc_xyz_inlier_mask = acc_zsc_xyz[np.abs(acc_zsc_xyz) > 3].groupby(level=0).count() == 0
        disp_mag_inlier_mask = disp_zsc_mag[np.abs(disp_zsc_mag) > 3].groupby(level=0).count() == 0
        disp_xyz_inlier_mask = disp_zsc_xyz[np.abs(disp_zsc_xyz) > 3].groupby(level=0).count() == 0

        inlier_mask = DataFrame(dtype=bool, index=self._acceleration_df.index, columns=self._acceleration_df.columns)
        combined_mask = acc_mag_inlier_mask & acc_xyz_inlier_mask & disp_mag_inlier_mask & disp_xyz_inlier_mask
        for idx in self._acceleration_df.index.values:
            inlier_mask.loc[idx, combined_mask.columns.values] = combined_mask.loc[idx[0]]

        return inlier_mask

    @staticmethod
    def _xyz_to_mag_df(dataframe_in: DataFrame, node_list: list[str] = None) -> DataFrame:
        """ Convert a dataframe generated in the xyz into a magnitude dataframe """
        if node_list is None:
            node_list = dataframe_in.index.levels[0]
        keys = dataframe_in.keys()
        dataframe_out = DataFrame(dtype=np.float64, index=node_list, columns=keys)

        for key in keys:
            for kp in node_list:
                xyz = dataframe_in.loc[kp, key]
                if not xyz.isnull().any():
                    dataframe_out.loc[kp, key] = np.linalg.norm(xyz).astype(np.float64)

        return dataframe_out

    def acceleration_xyz(self, clean:bool = False) -> DataFrame:
        if clean:
            return self._acceleration_df[self._inlier_mask]
        else:
            return self._acceleration_df

    def acceleration_mag(self, clean:bool = False) -> DataFrame:
        return self._xyz_to_mag_df(self.acceleration_xyz(clean), self._kp_in_df)

    def velocity_xyz(self, clean: bool = False):
        if clean:
            return self._velocity_df[self._inlier_mask]
        else:
            return self._velocity_df

    def velocity_mag(self, clean: bool = False) -> DataFrame:
        return self._xyz_to_mag_df(self.velocity_xyz(clean), self._kp_in_df)

    def displace_xyz(self, clean: bool = False) -> DataFrame:
        """ The displacement per frame if neighbouring frames contain non NaN positions given as a `MultiIndex` `DataFrame`
        with X,Y,Z as the second level Index"""
        if clean:
            return self._displace_df[self._inlier_mask]
        else:
            return self._displace_df

    def displace_mag(self, clean: bool = False) -> DataFrame:
        """ The magnitude displacement per frame if neighbouring frames contain non NaN positions given as a DataFrame """
        return self._xyz_to_mag_df(self.displace_xyz(clean), self._kp_in_df)

    def position_xyz(self, clean: bool = False) -> DataFrame:
        """ The position with respect to [0,0,0] given as X,Y,Z in a MultiIndex DataFrame"""
        if clean:
            return self._position_df[self._inlier_mask]
        else:
            return self._position_df

    def position_mag(self, clean: bool = False) -> DataFrame:
        """ The position offset from [0,0,0] given as a magnitude per frame in a DataFrame"""
        return self._xyz_to_mag_df(self.position_xyz(clean), self._kp_in_df)

    @property
    def kp_in_df(self) -> [str]:
        """ The names of the nodes contained within the Animal DataFrame given as a list of strings"""
        return self._kp_in_df

def calculate_heading(adf: AnimalDataFrame) -> DataFrame:
    """ Use the neck, eyes and antennae to get an approximate vector of direction for the head"""

    #Use filtered position

    #Maybe find the average relationship between the neck and each key point and use it as a way to guess the other positions

    # Get the vector offset for all neck pairings, remove outliers by length of vector
