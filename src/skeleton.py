import toml
from pathlib import Path

class SkeletonToml:
    animal: str
    link_name_list: list[str]
    skeleton_connectivity: list[int]

    def __init__(self, toml_path: Path):
        if toml_path is not Path:
            toml_path = Path(toml_path).resolve()
        with open(toml_path, 'r') as f:
            config = toml.load(f)
        self.animal = config['animal']
        self.link_name_list = config['link_name_list']
        self.skeleton_connectivity = config['skeleton_connectivity']
