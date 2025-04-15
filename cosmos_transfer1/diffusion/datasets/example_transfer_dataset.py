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

"""
Run this command to interactively debug:
PYTHONPATH=. python cosmos_transfer1/diffusion/datasets/example_transfer_dataset.py
"""

import os
import warnings
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from decord import VideoReader, cpu
import pickle

from cosmos_transfer1.diffusion.datasets.augmentor_provider import AUGMENTOR_OPTIONS
from cosmos_transfer1.diffusion.datasets.augmentors.control_input import VIDEO_RES_SIZE_INFO


CTRL_AUG_KEYS = {
    "depth": "depth",
    "seg": "segmentation",
    "human_kpts": "human_kpts",
}

# mappings between control types and corresponding sub-folders names in the data folder
CTRL_TYPE_INFO = {
    "human_kpts": {"folder": "human_kpts", "format": "pickle", "data_dict_key": "human_kpts"},
    "depth": {"folder": "depth", "format": "mp4", "data_dict_key": "depth"},
    "seg": {"folder": "seg", "format": "pickle", "data_dict_key": "segmentation"},
    "edge": {"folder": None},  # Canny edge, computed on-the-fly
    "vis": {"folder": None},   # Blur, computed on-the-fly
    "upscale": {"folder": None} # Computed on-the-fly
}


class ExampleTransferDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        chunk_size,
        num_frames,
        resolution,
        start_frame_interval=1,
        hint_key="control_input_vis",
        # augmentor_name="video_basic_augmentor",
        is_train=True
    ):
        """Dataset class for loading video-text-to-video generation data with control inputs.

        Args:
            dataset_dir (str): Base path to the dataset directory
            chunk_size (int): Interval between sampled frames in a sequence.
            num_frames (int): Number of frames to load per sequence
            resolution (str): resolution of the target video size
            start_frame_interval (int): Interval for starting frames
            hint_key (str): The hint key for loading the correct control input data modality
            is_train (bool): Whether this is for training

        NOTE: in our example dataset we do not have a validation dataset. The is_train flag is kept here for customized configuration.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.start_frame_interval = start_frame_interval
        self.chunk_size = chunk_size
        self.sequence_length = num_frames
        self.is_train = is_train
        self.resolution = resolution
        assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

        # Control input setup with file formats
        self.ctrl_type = hint_key.lstrip("control_input_")
        self.ctrl_data_pth_config = CTRL_TYPE_INFO[self.ctrl_type]

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

        augmentor_name = f"video_ctrlnet_augmentor_{hint_key}"
        # The augmentor will process the 'raw' control input data to the tensor,
        # add it to the data dict, and resize both the video and the control input to the model's required input size
        self.augmentor = AUGMENTOR_OPTIONS[augmentor_name](
            resolution=resolution,
            append_fps_frames=False
        )

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
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)

        # Check if all required control files exist
        ctrl_files_exist = True
        video_name = os.path.basename(video_path).replace(".mp4", "")

        # load control input file if needed
        if self.ctrl_data_pth_config["folder"] is not None:
            ctrl_path = os.path.join(
                self.dataset_dir,
                self.ctrl_data_pth_config["folder"],
                f"{video_name}.{self.ctrl_data_pth_config['format']}"
            )
            if not os.path.exists(ctrl_path):
                ctrl_files_exist = False
                warnings.warn(f"Missing control input file: {ctrl_path}")
        else:
            ctrl_files_exist = True

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

            if self.ctrl_data_pth_config["folder"] is not None:
                sample["ctrl_path"] = os.path.join(
                    self.dataset_dir,
                    self.ctrl_data_pth_config["folder"],
                    f"{video_name}.{self.ctrl_data_pth_config['format']}"
                )
            else:
                sample["ctrl_path"] = None

            sample["frame_ids"] = []
            sample["chunk_index"] = -1
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.chunk_size
            if len(sample["frame_ids"]) == self.sequence_length:
                sample["chunk_index"] += 1
                samples.append(sample)
        return samples

    def _load_control_data(self, sample):
        """Load control data for the video clip."""
        data_dict = {}
        frame_ids = sample["frame_ids"]
        ctrl_path = sample["ctrl_path"]
        try:
            if self.ctrl_type == "seg":
                with open(ctrl_path, 'rb') as f:
                    ctrl_data = pickle.load(f)
                # key should match line 982 at cosmos_transfer1/diffusion/datasets/augmentors/control_input.py
                data_dict["segmentation"] = ctrl_data
            elif self.ctrl_type == "human_kpts":
                with open(ctrl_path, 'rb') as f:
                    ctrl_data = pickle.load(f)
                data_dict["human_kpts"] = ctrl_data
            elif self.ctrl_type == "depth":
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                # Ensure the depth video has the same number of frames
                assert len(vr) >= frame_ids[-1] + 1, \
                    f"Depth video {ctrl_data} has fewer frames than main video"

                # Load the corresponding frames
                depth_frames = vr.get_batch(frame_ids).asnumpy()
                depth_frames = torch.from_numpy(depth_frames).permute(0, 3, 1, 2)  # [T,C,H,W]
                data_dict["depth"] = {
                    "video": depth_frames,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1],
                    "chunk_index": sample["chunk_index"]
                }

        except Exception as e:
            warnings.warn(f"Failed to load control data from {ctrl_data}: {str(e)}")
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
            data["chunk_index"] = sample["chunk_index"]
            data["image_size"] = torch.tensor([704, 1280, 704, 1280]).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 704, 1280).cuda()

            if self.ctrl_type:
                ctrl_data = self._load_control_data(sample)
                if ctrl_data is None:  # Control data loading failed, discard this sample and reload another sample
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


if __name__ == "__main__":
    dataset = ExampleTransferDataset(
        dataset_dir="assets/example_transfer_training_data/",
        hint_key="control_input_seg",
        chunk_size=1,
        num_frames=121,
        resolution="720",
        # augmentor_name="video_basic_augmentor",
        is_train=True
    )

    indices = [0, 13, 200, -1]
    for idx in indices:
        data = dataset[idx]
        print(
            (
                f"{idx=} "
                f"{data['video'].sum()=}\n"
                f"{data['video'].shape=}\n"
                f"{data['depth']['video'].sum()=}\n"
                f"{data['depth']['video'].shape=}\n"
                f"{data['video_name']=}\n"
                f"{data['t5_text_embeddings'].shape=}\n"
                "---"
            )
        )
