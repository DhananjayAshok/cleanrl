import os
import numpy as np
import torch
from time import perf_counter_ns


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
