from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import yaml


@dataclass
class LeRobotConfig:
    """Configuration for LeRobot Dataset Conversion."""

    # --- Required Fields ---
    repo_id: str
    """Hugging Face repository ID (e.g., 'username/dataset-name')."""

    input_path: Path
    """Path to input HDF5 file or directory."""

    # --- Optional Fields ---
    fps: int = 20
    """Frames per second for the dataset."""

    robot_type: Optional[str] = None
    """Type of robot (e.g., 'panda', 'ur5')."""

    push_to_hub: bool = False
    """Whether to push the dataset to the Hub after conversion."""

    private_repo: bool = False
    """Make the repository private when pushing."""

    tags: Optional[List[str]] = None
    """Optional tags for the Hugging Face Hub."""

    output_dir: Path = Path("./data/lerobot_formatted")
    """Directory to save the formatted dataset."""

    batch_size: int = 1
    """Batch size for processing frames."""

    task: str = "unknown"
    """Default task name if mapping is not found."""

    task_mapping: Optional[Union[Dict[str, str], List[str]]] = None
    """Mapping of files to tasks."""

    obs_mode: Literal["joint", "eef"] = "joint"
    """
    Format for 'observation.state':
    - 'joint': 7D (6 joints + gripper)
    - 'eef':   7D (3 pos + 3 axis-angle + gripper)
    """

    action_mode: Literal["joint", "eef"] = "joint"
    """
    Format for 'action':
    - 'joint': 7D (6 joints + gripper)
    - 'eef':   7D (3 pos + 3 axis-angle + gripper)
    
    Note: This assumes the HDF5 'actions' array matches this mode.
    """

    action_key: str = "actions"
    """HDF5 key to read actions from. Change this if your EEF actions are stored under a different name (e.g. 'actions_eef')."""

    @classmethod
    def from_yaml(cls, path: Path) -> "LeRobotConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert path strings to Path objects manually where needed
        if "input_path" in data:
            data["input_path"] = Path(data["input_path"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        # Tyro/Dataclasses will handle the rest, but we need to ensure
        # the dict matches the fields.
        # Note: If you want strict validation here, you can use dacite.
        return cls(**data)
