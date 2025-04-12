# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
import traceback
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from torchvision import transforms as T

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from decord import VideoReader, cpu
import pickle

from cosmos_transfer1.diffusion.datasets.dataset_utils import (
    ResizeSmallestSideAspectPreserving,
    CenterCrop,
    Normalize,
)
from cosmos_transfer1.diffusion.training.datasets.dataset_utils import (
    ToTensorVideo, Resize_Preprocess
)
from cosmos_transfer1.diffusion.datasets.augmentor_provider import AUGMENTOR_OPTIONS
from cosmos_transfer1.diffusion.datasets.augmentors.control_input import (
    VIDEO_RES_SIZE_INFO,
    AddControlInputComb,
    AddControlInput,
)
# mappings between control types and corresponding sub-folders names in the data folder
CTRL_AUG_KEYS = {
    "depth": "depth",
    "seg": "seg",
    "human_kpts": "human_kpts",
}

# Map control types to their folder names and whether they need pre-stored data
CTRL_TYPE_INFO = {
    "human_kpts": {"folder": "human_annotation", "needs_data": True},
    "depth": {"folder": "depth", "needs_data": True},
    "seg": {"folder": "seg", "needs_data": True},
    "canny": {"folder": None, "needs_data": False},  # Computed on-the-fly
    "blur": {"folder": None, "needs_data": False},   # Computed on-the-fly
    "upscale": {"folder": None, "needs_data": False} # Computed on-the-fly
}


@dataclass
class VideoDatasetWithCtrlConfig:  # TODO (qianlim) not needed?
    """Configuration for VideoDatasetWithCtrlAnnotations.

    Args:
        dataset_name (str): Name of the dataset (e.g. "hdvila:control_input_human_kpts")
        resolution (str): Data resolution ("256", "720", "1080")
        num_video_frames (int): Number of frames to sample
        video_decoder_name (str): Name of the video decoder
        is_train (bool): Whether in training mode
        use_fps_control (bool): Whether to use FPS control
        min_fps_thres (int): Minimum FPS threshold when use_fps_control is True
        max_fps_thres (int): Maximum FPS threshold when use_fps_control is True
        dataset_resolution (str, optional): Minimum resolution to use in dataset
        chunk_size (int, optional): Size of video chunks
        rename_keys_src (list): Source keys to rename
        rename_keys_target (list): Target keys to rename to
        blur_config (dict, optional): Configuration for blur control
    """
    dataset_name: str  # e.g. "hdvila:control_input_human_kpts"
    resolution: str
    num_video_frames: int
    is_train: bool
    video_decoder_name: str = "video_decoder_w_controlled_fps"
    use_fps_control: bool = False
    min_fps_thres: int = 4
    max_fps_thres: int = 24
    dataset_resolution: Optional[str] = None
    chunk_size: Optional[int] = None
    rename_keys_src: List[str] = field(default_factory=list)
    rename_keys_target: List[str] = field(default_factory=list)
    blur_config: Optional[dict] = None


class VideoDatasetWithCtrlAnnotations(Dataset):
    def __init__(
        self,
        dataset_dir,
        sequence_interval,
        num_frames,
        video_size,
        resolution,
        start_frame_interval=1,
        ctrl_types=None,
        augmentor_name="video_basic_augmentor",
        is_train=True
    ):
        """Dataset class for loading image-text-to-video generation data with control inputs.

        Args:
            dataset_dir (str): Base path to the dataset directory
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames
            start_frame_interval (int): Interval for starting frames
            ctrl_types (list): List of control types to load (e.g. ["human_kpts", "depth"])
            augmentor_name (str): Name of the augmentor to use
            is_train (bool): Whether this is for training
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.start_frame_interval = start_frame_interval
        self.sequence_interval = sequence_interval
        self.sequence_length = num_frames
        self.is_train = is_train
        self.resolution = resolution

        assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

        # Control input setup with file formats
        self.ctrl_types = ctrl_types or []
        self.ctrl_config = {
            "human_kpts": {"folder": "human_kpts", "format": "pkl"},
            "depth": {"folder": "depth", "format": "mp4"},
            "segmentation": {"folder": "seg", "format": "pkl"}
        }

        # Set up directories
        video_dir = os.path.join(self.dataset_dir, "videos")
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")
        print(f"{len(self.video_paths)} videos in total")

        # Initialize samples
        self.samples = self._init_samples(self.video_paths)
        self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))
        print(f"{len(self.samples)} samples in total")

        # Set up preprocessing and augmentation
        self.wrong_number = 0
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

        if self.ctrl_types:
            self.augmentor = AUGMENTOR_OPTIONS[augmentor_name](
                resolution=resolution,
                text_transform_input_keys="",
                append_fps_frames=False
            )
        else:
            self.augmentor = None

    def _init_samples(self, video_paths):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_video_path = {
                executor.submit(self._load_and_process_video_path, video_path): video_path
                for video_path in video_paths
            }
            for future in tqdm(as_completed(future_to_video_path), total=len(video_paths)):
                samples.extend(future.result())
        return samples

    def _load_and_process_video_path(self, video_path):
        # TODO (qianlim) add support for loading a chunck of N frames from loaded video and return the video and the frame ids
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)

        # Check if all required control files exist
        ctrl_files_exist = True
        video_name = os.path.basename(video_path).replace(".mp4", "")
        for ctrl_type in self.ctrl_types:
            if ctrl_type not in self.ctrl_config:
                continue
            ctrl_info = self.ctrl_config[ctrl_type]
            ctrl_path = os.path.join(
                self.dataset_dir,
                ctrl_info["folder"],
                f"{video_name}.{ctrl_info['format']}"
            )
            if not os.path.exists(ctrl_path):
                ctrl_files_exist = False
                warnings.warn(f"Missing control file: {ctrl_path}")
                break

        samples = []
        if not ctrl_files_exist:
            return samples

        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample["video_path"] = video_path
            sample["t5_embedding_path"] = os.path.join(
                self.t5_dir,
                os.path.basename(video_path).replace(".mp4", ".pickle"),
            )
            # Add control paths with their formats
            sample["ctrl_paths"] = {}
            for ctrl_type in self.ctrl_types:
                if ctrl_type in self.ctrl_config:
                    ctrl_info = self.ctrl_config[ctrl_type]
                    sample["ctrl_paths"][ctrl_info["folder"]] = {
                        "path": os.path.join(
                            self.dataset_dir,
                            ctrl_info["folder"],
                            f"{video_name}.{ctrl_info['format']}"
                        ),
                        "format": ctrl_info["format"]
                    }

            sample["frame_ids"] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval
            if len(sample["frame_ids"]) == self.sequence_length:
                samples.append(sample)
        return samples

    def _load_control_data(self, sample):
        """Load control data for the video clip."""
        data_dict = {}
        frame_ids = sample["frame_ids"]

        for ctrl_folder, ctrl_info in sample["ctrl_paths"].items():
            try:
                if ctrl_info["format"] == "pkl":
                    # Load pickle files (for human_kpts and segmentation)
                    with open(ctrl_info["path"], 'rb') as f:
                        ctrl_data = pickle.load(f)
                    data_dict[ctrl_folder] = ctrl_data

                elif ctrl_info["format"] == "mp4":
                    # Load video files (for depth)
                    vr = VideoReader(ctrl_info["path"], ctx=cpu(0))
                    # Ensure the depth video has the same number of frames
                    assert len(vr) >= frame_ids[-1] + 1, \
                        f"Depth video {ctrl_info['path']} has fewer frames than main video"

                    # Load the corresponding frames
                    depth_frames = vr.get_batch(frame_ids).asnumpy()
                    depth_frames = torch.from_numpy(depth_frames).permute(0, 3, 1, 2)  # [T,C,H,W]

                    data_dict[ctrl_folder] = {
                        "video": depth_frames,
                        "frame_start": frame_ids[0],
                        "frame_end": frame_ids[-1],
                        "chunk_index": 0  # Required by some augmentors
                    }

            except Exception as e:
                warnings.warn(f"Failed to load control data from {ctrl_info['path']}: {str(e)}")
                return None

        return data_dict

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            fps = 24
        return frame_data, fps

    def _get_frames(self, video_path, frame_ids):
        frames, fps = self._load_video(video_path, frame_ids)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    def __getitem__(self, index):
        try:
            sample = self.samples[index]
            video_path = sample["video_path"]
            frame_ids = sample["frame_ids"]

            data = dict()

            # Load video frames
            video, fps = self._get_frames(video_path, frame_ids)
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # Basic data
            data["video"] = video
            data["video_name"] = {
                "video_path": video_path,
                "t5_embedding_path": sample["t5_embedding_path"],
                "start_frame_id": str(frame_ids[0]),
            }

            # Load T5 embeddings
            with open(sample["t5_embedding_path"], "rb") as f:
                t5_embedding = pickle.load(f)[0]
            data["t5_text_embeddings"] = torch.from_numpy(t5_embedding).cuda()
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()

            # Add metadata
            data["fps"] = fps
            data["frame_start"] = frame_ids[0]
            data["frame_end"] = frame_ids[-1]
            data["image_size"] = torch.tensor([704, 1280, 704, 1280]).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 704, 1280).cuda()

            if self.ctrl_types:
                ctrl_data = self._load_control_data(sample)
                if ctrl_data is None:  # Control data loading failed
                    return self[np.random.randint(len(self.samples))]
                data.update(ctrl_data)

                # Apply augmentations including control input processing
                for aug_name, aug_fn in self.augmentor.items():
                    data = aug_fn(data)

            return data

        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['video_path']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"