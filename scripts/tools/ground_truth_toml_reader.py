from pathlib import Path
import toml
class GroundTruth:
    def __init__(self, path: [Path | str], prefix: str, session_number: [str | int]):

        if path is not Path:
            path = Path(path).resolve()

        file_path = path / f"{prefix}/inputs/ground_truth_ant_contact.toml"
        full_toml = toml.load(file_path)
        gt = full_toml[prefix][f'session{session_number}']
        output_dict = {}
        output_dict["enter"] = gt[0][0]
        output_dict["touch"] = gt[0][1]
        output_dict["grasp"] = gt[0][2]
        output_dict["exit"] = gt[0][3]
        output_dict["tracks"] = gt[1]
        output_dict["reliable_track"] = gt[2]

        self.__dict__ = output_dict

    def __getitem__(self, item):
        return self.__dict__[item]