import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from .config import LeRobotConfig
from .utils import quat2axisangle

logger = logging.getLogger(__name__)


class LeRobotDatasetConverter:
    def __init__(self, config: LeRobotConfig) -> None:
        self.cfg = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path: Path = self.cfg.output_dir / f"conversion_{timestamp}"

        # 1. Image Features (Constant)
        self.features: dict[str, dict[str, Any]] = {
            "observation.images.camera_base": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "rgb"],
            },
            "observation.images.camera_wrist_right": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "rgb"],
            },
        }

        # 2. Configure Observation Space (Input)
        if self.cfg.obs_mode == "eef":
            logger.info("Observation Mode: EEF (Pose + Gripper)")
            self.features["observation.state"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "eef_pose": [
                        "pos_x",
                        "pos_y",
                        "pos_z",
                        "rot_x",
                        "rot_y",
                        "rot_z",
                        "gripper_width",
                    ]
                },
            }
        else:
            logger.info("Observation Mode: Joint (Angles + Gripper)")
            self.features["observation.state"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "motors": [
                        "joint0",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "gripper",
                    ]
                },
            }

        # 3. Configure Action Space (Output)
        if self.cfg.action_mode == "eef":
            logger.info(
                f"Action Mode: EEF (Pose Delta) reading from '{self.cfg.action_key}'"
            )
            self.features["action"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "eef_action": [
                        "dx",
                        "dy",
                        "dz",
                        "dr_x",
                        "dr_y",
                        "dr_z",
                        "gripper_action",
                    ]
                },
            }
        else:
            logger.info(
                f"Action Mode: Joint (Velocity/Pos) reading from '{self.cfg.action_key}'"
            )
            self.features["action"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "motors": [
                        "joint0",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "gripper",
                    ]
                },
            }

    # ... [create_dataset and get_task_for_file methods unchanged] ...

    def create_dataset(self) -> LeRobotDataset:
        return LeRobotDataset.create(
            repo_id=self.cfg.repo_id,
            fps=self.cfg.fps,
            features=self.features,
            root=self.output_path,
            robot_type=self.cfg.robot_type,
            use_videos=True,
            image_writer_processes=16,
            image_writer_threads=20,
            batch_encoding_size=self.cfg.batch_size,
        )

    def get_task_for_file(self, file_path: Path, file_index: int) -> str:
        mapping = self.cfg.task_mapping
        if mapping is None:
            return self.cfg.task
        if isinstance(mapping, list):
            return mapping[file_index] if file_index < len(mapping) else self.cfg.task
        if isinstance(mapping, dict):
            return mapping.get(file_path.name, self.cfg.task)
        return self.cfg.task

    def process_episode(
        self, hdf5_path: Path, dataset: LeRobotDataset, file_task: str
    ) -> int:
        processed_count = 0
        try:
            with h5py.File(hdf5_path, "r") as hdf5_file:
                if "data" not in hdf5_file:
                    return 0
                data_group = hdf5_file["data"]

                for demo_name in sorted(
                    [k for k in data_group.keys() if k.startswith("demo_")]
                ):
                    try:
                        demo_group = data_group[demo_name]
                        obs_group = demo_group["obs"]
                        task = hdf5_file.attrs.get(
                            "task", demo_group.attrs.get("task", file_task)
                        )

                        # --- 1. Extract Observation (Input) ---
                        robot_state: np.ndarray
                        if self.cfg.obs_mode == "eef":
                            # EEF Mode: Pos (3) + AxisAngle (3) + Gripper (1)
                            eef_pos = obs_group["robot0_eef_pos"][:].astype(np.float32)
                            eef_quat = obs_group["robot0_eef_quat"][:].astype(
                                np.float32
                            )
                            eef_axis_angle = np.array(
                                [quat2axisangle(q) for q in eef_quat], dtype=np.float32
                            )

                            # Gripper
                            gripper_qpos = obs_group["robot0_gripper_qpos"][:].astype(
                                np.float32
                            )
                            robot_state = np.concatenate(
                                [eef_pos, eef_axis_angle, gripper_qpos[:, :1]], axis=1
                            )
                        else:
                            # Joint Mode: Joints (6) + Gripper (1)
                            joint_pos = obs_group["robot0_joint_pos"][:].astype(
                                np.float32
                            )
                            gripper_qpos = obs_group["robot0_gripper_qpos"][:].astype(
                                np.float32
                            )
                            robot_state = np.concatenate(
                                [joint_pos, gripper_qpos[:, :1]], axis=1
                            )

                        # --- 2. Extract Action (Output) ---
                        # Note: We assume the HDF5 'actions' are already in the correct format
                        # matching the mode, OR the user provided a specific key (e.g. 'actions_eef')
                        if self.cfg.action_key not in demo_group:
                            logger.warning(
                                f"Action key '{self.cfg.action_key}' not found in {demo_name}. Skipping."
                            )
                            continue

                        actions = demo_group[self.cfg.action_key][:].astype(np.float32)

                        # --- 3. Process Images & Save ---
                        agentview_img = obs_group["agentview_image"][:]
                        eye_in_hand_img = obs_group["robot0_eye_in_hand_image"][:]

                        min_length = min(
                            len(actions), len(robot_state), len(agentview_img)
                        )

                        for i in range(min_length):
                            dataset.add_frame(
                                {
                                    "observation.images.camera_base": torch.from_numpy(
                                        agentview_img[i]
                                    ),
                                    "observation.images.camera_wrist_right": torch.from_numpy(
                                        eye_in_hand_img[i]
                                    ),
                                    "observation.state": torch.from_numpy(
                                        robot_state[i]
                                    ),
                                    "action": torch.from_numpy(actions[i]),
                                    "task": task,
                                }
                            )

                        dataset.save_episode()
                        processed_count += 1
                        logger.info(
                            f"Processed {demo_name} ({min_length} frames) | Obs: {self.cfg.obs_mode} | Act: {self.cfg.action_mode}"
                        )

                    except Exception as e:
                        logger.error(f"Error processing {demo_name}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error reading file {hdf5_path}: {e}")

        return processed_count

    def run_conversion(self) -> None:
        if self.cfg.input_path.is_file():
            files = [self.cfg.input_path]
        elif self.cfg.input_path.is_dir():
            files = sorted(self.cfg.input_path.rglob("*.hdf5"))
        else:
            raise FileNotFoundError(f"{self.cfg.input_path} not found")

        dataset = self.create_dataset()
        total = 0
        for i, f in enumerate(tqdm(files)):
            total += self.process_episode(f, dataset, self.get_task_for_file(f, i))

        dataset.finalize()
        if self.cfg.push_to_hub:
            dataset.push_to_hub(private=self.cfg.private_repo, tags=self.cfg.tags)
