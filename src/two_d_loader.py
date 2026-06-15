import logging
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# Used code from Florent Le Moel in FlorentLM/mokap:src/fileio.py

def merge_pandas_dfs(list_of_dfs, reset_tracks=True):
    list_of_dfs = list_of_dfs.copy()

    if reset_tracks:
        last_nb_tracks = 0
        for df in list_of_dfs:
            track_ids = df[('comments', 'instance')].factorize()[0] + last_nb_tracks
            last_nb_tracks += np.unique(track_ids).shape[0]
            df['track'] = track_ids

    multiview_df = pd.concat(list_of_dfs, join='outer')

    if reset_tracks:
        if 'track' in multiview_df.index.names:
            multiview_df = multiview_df.reset_index('track', drop=True)  # Reset tracks: get rid of the old ones

    if set(multiview_df.index.names) == {None}:
        multiview_df = multiview_df.set_index(['camera', 'track', 'frame'])
    else:
        multiview_df = multiview_df.reset_index().set_index(['camera', 'track', 'frame'])

    # Set the cameras level as a categorical
    multiview_df.index = multiview_df.index.set_levels(
        pd.CategoricalIndex(multiview_df.index.levels[0],
                            categories=sorted(multiview_df.index.levels[0]), ordered=True), level=0)

    # And apply the sorted categorical index for the cameras
    multiview_df = multiview_df.sort_index()

    return multiview_df

# Used code from Florent Le Moel in FlorentLM/mokap:src/fileio.py
def SLP_to_pandas(slp_content, camera_name=None, session=None):
    def instance_to_row(instance, is_manual):

        original_track = instance.track.name if instance.track else ''
        instance_score = float(instance.score) if hasattr(instance, 'score') else int(is_manual)
        tracking_score = float(instance.tracking_score) if hasattr(instance, 'tracking_score') else int(is_manual)

        values = []
        for i, node in enumerate(instance.skeleton.nodes):
            # if not instance.points[node].visible:
            if not instance.points[i]['visible']:
                x = np.nan
                y = np.nan
                s = 0.0
            else:
                # x = float(instance.points[node].x) if hasattr(instance.points[node], 'x') else np.nan
                # y = float(instance.points[node].y) if hasattr(instance.points[node], 'y') else np.nan
                # s = float(instance.points[node].score) if hasattr(instance.points[node], 'score') else 1.0
                try:
                    x, y = instance.points[i]['xy']
                    s = instance.points[i]['score']
                except:
                    x = y = np.nan
                    s = 1.0
                x, y, s = float(x), float(y), float(s)

            values.extend([x, y, s])
        return values + [instance_score, tracking_score, original_track]

    keypoints = slp_content.skeleton.node_names
    columns = (['camera.', 'frame.']
               + [f"{k}.{a}" for k in keypoints for a in ['x', 'y', 'score']]
               + ['comments.instance_score', 'comments.tracking_score', 'comments.instance'])

    rows = []
    for frame_content in slp_content.labeled_frames:
        source_video = Path(frame_content.video.filename)
        if camera_name is None or camera_name in source_video.stem:  # if name is not passed, assume we load everything
            if session is None or str(session) in source_video.stem:
                for i, instance in enumerate(frame_content.instances):
                    is_manual = instance in frame_content.user_instances
                    row = instance_to_row(instance, is_manual)
                    if row[-1] == '':
                        row[-1] = f'instance_{i}'
                    if session is not None:
                        row[-1] = f"{session}_{row[-1]}"  # prepend session in the track nb
                    row = [camera_name, frame_content.frame_idx + 1] + row
                    rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.columns = pd.MultiIndex.from_tuples([col.split('.') for col in df.columns])

    return df

def SLP_to_polars(slp_content, camera_name: str, session: str) -> pl.DataFrame:

    keypoint_names = slp_content.skeleton.node_names
    rows = []

    for frame_content in slp_content.labeled_frames:
        source_video = Path(frame_content.video.filename)
        if camera_name not in source_video.stem or str(session) not in source_video.stem:
            continue

        frame_idx = frame_content.frame_idx
        for i, instance in enumerate(frame_content.instances):

            is_manual = instance in frame_content.user_instances
            track_name = instance.track.name if instance.track else f'instance_{i}'

            for kp_idx, node in enumerate(instance.skeleton.nodes):
                point_data = instance.points[kp_idx]

                x, y = np.nan, np.nan

                # check point visibility
                if not 'visible' in point_data.dtype.names or not point_data['visible']:
                    score = 0.0
                else:
                    # point visible, get coordinates
                    if 'xy' in point_data.dtype.names:
                        x, y = point_data['xy']

                    # score presence and manual annotation status
                    if 'score' in point_data.dtype.names:
                        score = point_data['score']
                    else:
                        # if score is missing, its value depends on whether it's a manual annotation or not
                        score = 1.0 if is_manual else 0.0

                rows.append({
                    "camera": camera_name,
                    "frame": frame_idx,
                    "track_id": track_name,
                    "keypoint": keypoint_names[kp_idx],
                    "x": float(x),
                    "y": float(y),
                    "score": float(score)
                })

    if not rows:
        return pl.DataFrame()

    return pl.from_dicts(rows)

def read_SLEAP(slp_path, to_polars=True):
    import sleap_io

    slp_path = Path(slp_path)
    slp_content = sleap_io.load_file(slp_path.as_posix())

    list_of_dfs = []

    source_files = [Path(v.filename) for v in slp_content.videos]
    cameras_names = set(f.stem.split('_')[-2] for f in source_files)
    sessions = set(f.stem.split('_')[-1] for f in source_files) #Is this just the number or "session11"

    for session in sessions:
        for cam_name in cameras_names:
            if to_polars:
                df = SLP_to_polars(slp_content, cam_name, session)
            else:
                df = SLP_to_pandas(slp_content, cam_name,
                                   session)  # This particular camera / session might not exist, so
            if not len(df) == 0:  # in that case the df is empty, we just skip it
                list_of_dfs.append(df)

    if to_polars:
        return pl.concat(list_of_dfs) if list_of_dfs else pl.DataFrame()
    else:
        return merge_pandas_dfs(list_of_dfs)

def catar_to_pandas(catar_content, camera_name=None, session=None):
    def instance_to_row(instance, is_manual):

        original_track = instance.track.name if instance.track else ''
        instance_score = float(instance.score) if hasattr(instance, 'score') else int(is_manual)
        tracking_score = float(instance.tracking_score) if hasattr(instance, 'tracking_score') else int(is_manual)

        values = []
        for i, node in enumerate(instance.skeleton.nodes):
            # if not instance.points[node].visible:
            if not instance.points[i]['visible']:
                x = np.nan
                y = np.nan
                s = 0.0
            else:
                # x = float(instance.points[node].x) if hasattr(instance.points[node], 'x') else np.nan
                # y = float(instance.points[node].y) if hasattr(instance.points[node], 'y') else np.nan
                # s = float(instance.points[node].score) if hasattr(instance.points[node], 'score') else 1.0
                try:
                    x, y = instance.points[i]['xy']
                    s = instance.points[i]['score']
                except:
                    x = y = np.nan
                    s = 1.0
                x, y, s = float(x), float(y), float(s)

            values.extend([x, y, s])
        return values + [instance_score, tracking_score, original_track]

    keypoints = np.unique(catar_content.keypoint)
    frames = np.unique(catar_content.frame)
    # keypoints = catar_content.skeleton.node_names
    columns = (['camera.', 'frame.']
               + [f"{k}.{a}" for k in keypoints for a in ['x', 'y', 'score']]
               + ['comments.instance_score', 'comments.tracking_score', 'comments.instance'])

    df = pd.DataFrame(0, columns=columns, index=frames)
    df.columns = pd.MultiIndex.from_tuples([col.split('.') for col in df.columns])
    df['camera'] = camera_name
    df['frame'] = frames

    for value in catar_content.itertuples():
        camera_name_value = value.camera_name.split('_')[1]
        if camera_name is None or camera_name in camera_name_value:
            session_name_value = value.file.split('-')[2]
            if session is None or session in session_name_value:
                df.at[value.frame,(f"{value.keypoint}", "x")] = int(value.x)
                df.at[value.frame,(f"{value.keypoint}", "y")] = int(value.y)
                df.at[value.frame,(f"{value.keypoint}", "score")] = int(1)

    return df


def read_catar(catar_path_in):
    catar_path = Path(catar_path_in)

    catar_content = pd.read_csv(catar_path.as_posix())

    list_of_dfs = []

    source_files = set(Path(v) for v in catar_content.file) #Most likely to cause issues with file not found
    cameras_names = set(s.split('_')[1] for s in catar_content.camera_name)
    sessions = set(f.stem.split('-')[-1] for f in source_files) #May be an issue with the text not just the number

    for session in sessions:
        for cam_name in cameras_names:

            df = catar_to_pandas(catar_content, cam_name,
                                   session)  # This particular camera / session might not exist, so
            if not len(
                    df) == 0:  # in that case the df is empty, we just skip it
                list_of_dfs.append(df)


    return merge_pandas_dfs(list_of_dfs)


def load_session(path, session='', use_polars=True):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Can't find {path.stem}!")

    if path.is_file():
        parent_folder = path.parent
        session = path.name.split('.')[0].split('_')[-1]
    else:
        parent_folder = path

    files_match = sorted(
        p.resolve() for p in parent_folder.glob(f'**/*session{session}*') if
        p.suffix in {'.csv', '.slp'})
    if not files_match:
        raise FileNotFoundError(
            f"Can't find any tracking result files for session '{session}' in {parent_folder}!")

    dfs = []
    loaded_slp, loaded_csv, loaded_catar_csv = 0, 0, 0

    for f in files_match:

        if f.suffix == '.slp' and 'predictions' in f.stem:
            dfs.append(read_SLEAP(f, to_polars=use_polars))
            loaded_slp += 1

        if f.suffix == '.csv' and 'predictions' in f.stem:
            if use_polars:
                dfs.append(pl.read_csv(f, separator=','))
            else:
                dfs.append(pd.read_csv(f, sep=','))
            loaded_csv += 1

        if f.suffix == '.csv' and 'catar' in f.stem:
            dfs.append(read_catar(f))
            loaded_catar_csv += 1


    if loaded_slp + loaded_csv + loaded_catar_csv == 0:
        print(f'No files loaded...')
        return []
    else:
        slp_txt = f'{loaded_slp} SLEAP slp' if loaded_slp > 0 else ''
        csv_txt = f'{loaded_csv} SLEAP csv' if loaded_csv > 0 else ''
        catar_txt = f'{loaded_catar_csv} CATAR csv' if loaded_catar_csv > 0 else ''
        and_txt = ' and ' if (loaded_slp > 0 and loaded_csv > 0) else ''
        and_catar_txt = ' and ' if ((loaded_slp > 0 or loaded_csv > 0) and loaded_catar_csv > 0) else ''
        print(f'Loaded {slp_txt}{and_txt}{csv_txt}{and_catar_txt}{catar_txt} files.')

        merged_df = merge_pandas_dfs(dfs, reset_tracks=True)

    return merged_df

class TwoDAnimalLoader:

    def __init__(self,
                 data_folder: Optional[str | Path] = None,
                 skeleton_toml_path: Optional[str | Path] = None,
                 animal_pkl: Optional[str | Path] = None,
                 session_number: Optional[int] = None,
                 track_number: Optional[list[int] | list[str]] = None,
                 prefix: Optional[str] = None,
                 ):

        self._animal_list = None
        if data_folder is not Path:
            data_folder = Path(data_folder).resolve()

        if skeleton_toml_path is None:
            raise Exception("No skeleton toml included")
        else:
            if animal_pkl is None:
                if data_folder is not None and session_number is not None and prefix is not None:
                    animal_pkl = data_folder / prefix / "outputs" / "tracking" / f"linked_tracks_session{session_number}.pkl"
                else:
                    raise Exception("No animal pkl included")




