# Class to process an AnimalStruct object and the DataFrame associated with it to be used in plotting.
import logging
from src.animal import AnimalStruct
import numpy as np
from pandas import DataFrame

#For post processing
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import scipy.optimize as opt
from filterpy.kalman import KalmanFilter
from alive_progress import alive_bar

logger = logging.getLogger(__name__)

class AnimalDataFrame:
    def __init__(self,
                 animal: AnimalStruct,
                 frames: list[int] = None,
                 node_list: list[str] = None,
                 calculate_inlier: bool = True,
                 signed: bool = True):
        """
        :param animal: The AnimalStruct object to use
        :param frames: A sequential frame window to query
        :param node_list: A list of nodes contained in the animal to query
        :param calculate_inlier: Whether to exclude outliers
        """

        #Pre-load all rays if they have not been loaded - used for ray/bone names
        all_frames = animal.get_frame_range()
        for all_frame in range(*all_frames):
            animal.get_pose_ray(all_frame)

        self._animal = animal
        self._frames = frames
        self._node_list = node_list
        self._signed = signed



        self._position_df, self._displace_df, self._velocity_df, self._acceleration_df, self._kp_in_df = self._get_motion_df()
        if calculate_inlier:
            self._inlier_mask = self._calculate_inlier()
        else:
            self._inlier_mask = None

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

        # acc_mag_df = self.acceleration_mag(clean=False) #Now has the option to get a signed plot
        # acc_zsc_mag = acc_mag_df.apply(modified_z_score, axis='columns', result_type='broadcast')
        # acc_zsc_xyz = self._acceleration_df.apply(modified_z_score, axis='columns', result_type='broadcast')

        disp_mag_df = self._xyz_to_mag_df(self._displace_df, self._kp_in_df)
        disp_zsc_mag = disp_mag_df.apply(modified_z_score, axis='columns', result_type='broadcast')
        disp_zsc_xyz = self._displace_df.apply(modified_z_score, axis='columns', result_type='broadcast')

        # acc_mag_inlier_mask = acc_zsc_mag[np.abs(acc_zsc_mag) > 2.5].groupby(level=0).count() == 0
        # acc_xyz_inlier_mask = acc_zsc_xyz[np.abs(acc_zsc_xyz) > 2.5].groupby(level=0).count() == 0
        disp_mag_inlier_mask = disp_zsc_mag[np.abs(disp_zsc_mag) > 3].groupby(level=0).count() == 0
        disp_xyz_inlier_mask = disp_zsc_xyz[np.abs(disp_zsc_xyz) > 3].groupby(level=0).count() == 0

        inlier_mask = DataFrame(dtype=bool, index=self._acceleration_df.index, columns=self._acceleration_df.columns)
        combined_mask = disp_mag_inlier_mask & disp_xyz_inlier_mask
        # combined_mask = acc_mag_inlier_mask & acc_xyz_inlier_mask & disp_mag_inlier_mask & disp_xyz_inlier_mask
        for idx in self._acceleration_df.index.values:
            inlier_mask.loc[idx, combined_mask.columns.values] = combined_mask.loc[idx[0]]

        return inlier_mask

    def post_process(self) -> DataFrame:
        """ Carry out various post-processing steps on the filtered position """

        def _interpolate_missing(keypoints, kp_in_df):
            keypoints_out = keypoints.copy()
            for joint in kp_in_df:
                for dim in ['x', 'y', 'z']:
                    # Extract the trajectory for this joint and dimension
                    traj = keypoints.loc[joint, dim]
                    # Find valid indices (not NaN)
                    not_nan_traj = traj.loc[traj.notna()]
                    if len(not_nan_traj.index) > 0:
                        # Interpolate missing values
                        interp_fn = interp1d(
                            not_nan_traj.index.values,  # Valid frame indices
                            not_nan_traj.values,  # Valid values
                            kind='linear',
                            fill_value="extrapolate"
                        )
                        traj = interp_fn(traj.index.values)
                    keypoints_out.loc[joint, dim] = traj
            return keypoints_out

        def _smooth_trajectories(keypoints, kp_in_df, window_length=5, polyorder=2):
            keypoints_out = keypoints.copy()
            for joint in kp_in_df:
                for dim in ['x', 'y', 'z']:
                    keypoints_out.loc[joint, dim] = savgol_filter(
                        keypoints.loc[joint, dim],
                        window_length=window_length,
                        polyorder=polyorder
                    )
            return keypoints_out

        def _enforce_bone_lengths(keypoints, bone_dict, kp_in_df):
            def _loss(x, target, bone_pairs, name_list):

                error = 0.0
                # Penalize deviation from observed (smoothed) keypoints
                error += np.sum((x - target) ** 2)
                # Penalize bone-length violations
                x_reshape = x.reshape((-1, 3))  # Reshape to (num_joints, 3)
                for bone in bone_pairs:
                    parent_idx = name_list.index(bone_pairs[bone]['parent'])
                    child_idx = name_list.index(bone_pairs[bone]['child'])
                    current_length = np.linalg.norm(x_reshape[parent_idx] - x_reshape[child_idx])
                    error += 10.0 * (current_length - bone_pairs[bone]['length']) ** 2  # Weighted term
                return error

            optimized = keypoints.copy()
            with alive_bar(title=f" Optimising Bone Length", total=len(optimized.columns), force_tty=True) as bar:
                for frame in optimized:
                    x0 = keypoints.loc[kp_in_df, frame]  # Initial guess
                    res = opt.minimize(
                        _loss,
                        x0,
                        args=(keypoints.loc[kp_in_df, frame], bone_dict, kp_in_df),
                        method='L-BFGS-B'  # Efficient for smooth problems
                    )
                    optimized.loc[kp_in_df, frame] = res.x
                    bar()
            return optimized

        def kalman_smooth_trajectories(keypoints, kp_in_df, process_noise=1e-3, measurement_noise=1e-2):
            keypoints_smoothed = keypoints.copy()
            for joint in kp_in_df:
                for dim in ['x', 'y', 'z']:
                    kf = KalmanFilter(dim_x=2, dim_z=1)  # Position and velocity
                    kf.x = np.array([keypoints.loc[joint, dim].iloc[0], 0])  # Initial state (pos, vel)
                    kf.F = np.array([[1., 1.], [0., 1.]])  # State transition (constant velocity)
                    kf.H = np.array([[1., 0.]])  # Measurement function
                    kf.P *= 1.0  # Covariance matrix
                    kf.R = measurement_noise  # Measurement noise
                    kf.Q = np.eye(2) * process_noise  # Process noise

                    # Filter forward
                    filtered = []
                    for p in keypoints.loc[joint, dim]:
                        kf.predict()
                        kf.update(p)
                        filtered.append(kf.x[0])

                    # Smooth backward (RTS smoother)
                    keypoints_smoothed.loc[joint, dim] = filtered

            return keypoints_smoothed



        keypoints_3d = self.position_xyz(clean=True)
        keypoints_3d_interpolated = _interpolate_missing(keypoints_3d, self._kp_in_df)
        keypoints_smoothed = _smooth_trajectories(keypoints_3d_interpolated, self._kp_in_df)

        edges = self._animal.ray_names
        #Add some other fixed length links
        additional_pair = [('eye_L', 'eye_R'), ('a_L0', 'a_R0'), ('m_L0', 'm_R0')]
        for pair in additional_pair:
            bone_name = pair[0] + '_' + pair[1]
            edges[bone_name] = {'parent': pair[0], 'child': pair[1]}

        for bone in edges:
            parent = edges[bone]['parent']
            child = edges[bone]['child']
            diff = keypoints_3d.loc[parent] - keypoints_3d.loc[child]
            length = np.nanmedian(diff.apply(np.linalg.norm), axis=0)
            edges[bone]['length'] = length

        keypoints_optimized = _enforce_bone_lengths(keypoints_smoothed, edges, self._kp_in_df)
        keypoints_final = kalman_smooth_trajectories(keypoints_optimized, self._kp_in_df)

        return keypoints_final

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
        if clean and self._inlier_mask is not None:
            return self._acceleration_df[self._inlier_mask]
        else:
            return self._acceleration_df

    def acceleration_mag_from_velocity(self, clean:bool = False) -> DataFrame:
        velocity_df = self.velocity_mag(clean=clean)
        keys = velocity_df.keys()
        node_list = velocity_df.index.values
        acceleration_df = DataFrame(dtype=np.float64, index=node_list, columns=keys)
        #Set the initial vel
        for key in keys:
            for kp in node_list:

                if velocity_df.keys().__contains__(key - 1):
                    u = velocity_df.loc[kp, key - 1]
                    dt = 1
                else:
                    u = np.nan
                    dt = np.nan

                v = velocity_df.loc[kp, key]

                a = np.asarray((v - u) / dt, dtype=np.float64)

                acceleration_df.loc[kp, key] = a

        return acceleration_df

    def acceleration_mag(self, clean:bool = False) -> DataFrame:
        if self._signed:
            return self.acceleration_mag_from_velocity(clean=clean)
        else:
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
        if clean and self._inlier_mask is not None:
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

    kp_interest = ['m_L0, m_R0, eye_L', 'eye_R', 'neck']
    #Use filtered position
    position_df = adf.position_xyz(clean=True)

    #Maybe find the average relationship between the neck and each key point and use it as a way to guess the other positions
    position_df.loc[['a_R0', 'a_L0', 'm_R0', 'm_L0', 'eye_L', 'eye_R']] - position_df.loc['neck']

    # Get the vector offset for all neck pairings, remove outliers by length of vector
