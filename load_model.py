#from extract_layer import *
import torch
from collections import OrderedDict


def key_transformation(key):
    return 'backbone.' + key


def rename_state_dict_keys(source, key_transformation, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)

    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.

    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"

        return old_key

    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if 'features' in key:
            new_key = key_transformation(key)
            new_state_dict[new_key] = value

    torch.save(new_state_dict, target)


path = "/home/yf/disk/pretrained/alexnet-owt-7be5be79.pth"
target_source = "/home/yf/disk/pretrained/alexnet.pth"
rename_state_dict_keys(path, key_transformation, target=None)
