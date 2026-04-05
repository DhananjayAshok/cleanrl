import os
import numpy as np
import torch
from time import perf_counter_ns
from gameboy_worlds.utils import show_frames

FRAME_STACK = 2


class Profiler:
    last_event = None
    last_event_time = None

    @staticmethod
    def event(name):
        current_time = perf_counter_ns()
        if Profiler.last_event is not None:
            elapsed_time = (current_time - Profiler.last_event_time) / 1e6
            print(f"{Profiler.last_event}->{name} | {elapsed_time:.2f} ms")
        Profiler.last_event = name
        Profiler.last_event_time = current_time


class MaxLengthList:
    def __init__(self, max_length):
        self.max_length = max_length
        self.data = []

    def insert(self, item, index):
        if index >= self.max_length:
            raise IndexError(
                f"Index {index} out of bounds for MaxLengthList with max_length {self.max_length}"
            )
        self.data.insert(index, item)
        if len(self.data) > self.max_length:
            self.data.pop(-1)

    def get_insert_index(self, item):
        """
        Get the index where the item should be inserted to preserve a descending sorted order
        """
        for i, existing_item in enumerate(self.data):
            if item > existing_item:
                return i
        if len(self.data) < self.max_length:
            return len(self.data)
        return (
            None  # item is not greater than any existing item and list is at max length
        )

    def do_item_insert(self, item):
        index = self.get_insert_index(item)
        if index is not None:
            self.insert(item, index)
            return index
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"MaxLengthList(max_length={self.max_length}, data={self.data})"

    def __str__(self):
        return self.__repr__()


def save_model(model_data, reward, model_save_folder):
    os.makedirs(model_save_folder, exist_ok=True)
    model_save_path = os.path.join(model_save_folder, "model.pt")
    torch.save(model_data, model_save_path)
    print(
        f"model saved to {model_save_path}, achieving reward {reward} (final will always be None)"
    )
    # save a reward text file as well for easy reference
    if reward is not None:
        if isinstance(reward, np.ndarray) or isinstance(reward, torch.Tensor):
            reward = reward.item()
        with open(os.path.join(model_save_folder, "train_reward.txt"), "w") as f:
            f.write(f"{reward}")


def save_ranked_models(model_data_list, rewards_list, model_save_folder):
    for i, (model_data, reward) in enumerate(zip(model_data_list, rewards_list)):
        save_model(model_data, reward, os.path.join(model_save_folder, f"rank_{i+1}"))


def save_all_models(final_model_data, model_data_list, rewards_list, model_save_folder):
    if model_save_folder is None:
        print(f"Warning: model_save_folder is None. Models will not be saved.")
        return
    save_model(final_model_data, None, os.path.join(model_save_folder, f"final"))
    save_ranked_models(
        model_data_list, rewards_list, model_save_folder=model_save_folder
    )


def depathify(string):
    return string.replace("/", "_").replace("\\", "_").replace(" ", "_")


def correct_torch_frame(frame: torch.Tensor):
    if frame.ndim == 2:  # then (H, W), add channel dimension
        frame = frame.unsqueeze(0)
    elif (
        frame.ndim == 3 and frame.shape[0] == 1
    ):  # then (C, H, W) but C is 1,reshape to (H, W, 1)
        frame = frame.permute(1, 2, 0)
    elif (
        frame.ndim == 3 and frame.shape[2] == 1
    ):  # then (H, W, C) but C is 1, do nothing
        pass
    return frame


def show_torch_frames(frames: torch.Tensor, titles=None, save=False):
    if isinstance(frames, list):
        frames = [frame for frame in frames]
    elif isinstance(frames, torch.Tensor):
        if frames.ndim == 4:  # (N, C, H, W) or (N, H, W, C)
            frames = [correct_torch_frame(frame) for frame in frames]
        elif frames.ndim == 3:  # (C, H, W), (H, W, C) or (N, H, W)
            # must detect which one it is
            if frames.shape[0] == 1 or frames.shape[0] == 3:  # (C, H, W)
                frames = [correct_torch_frame(frames)]
            else:
                frames = [correct_torch_frame(frame) for frame in frames]
        elif frames.ndim == 2:  # (H, W)
            frames = [correct_torch_frame(frames)]
        else:
            raise ValueError(
                f"Unsupported frame shape {frames.shape}, expected (N, C, H, W), (N, H, W, C), (C, H, W), (H, W, C) or (H, W)"
            )
    else:
        raise ValueError(
            f"Unsupported frames type {type(frames)}, expected list of torch.Tensor or torch.Tensor"
        )
    frames = [frame.detach().cpu().numpy() for frame in frames]
    return show_frames(frames, titles=titles, save=save)
