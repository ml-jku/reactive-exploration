import os
import yaml
import numpy as np
from gym import spaces
from stable_baselines3.common.preprocessing import is_image_space


def yaml_save(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)


def get_unique_params(hp_dict, combination_dict):
    # get keys that have multiple values
    keys = []
    ret = ""
    for key in hp_dict:
        if len(hp_dict[key]) > 1:
            keys += [key]
    # iterate over combination dict with these keys
    for key in keys:
        if isinstance(combination_dict[key], dict):
            ret += key + "="
            for key2 in combination_dict[key]:
                # append key:value to string
                val = combination_dict[key][key2]
                if type(val) is float:
                    val = '{0:.4f}'.format(val)
                ret += key2 + "=" + str(val) + "_"
        else:
            # append key:value to string
            val = combination_dict[key]
            if type(val) is float:
                val = '{0:.4f}'.format(val)
            ret += key + "=" + str(val) + "_"

    # remove last _
    if len(ret) > 0 and ret[-1] == '_':
        ret = ret[:-1]
    return ret


def transpose_image(image: np.ndarray) -> np.ndarray:
    """
    Transpose an image or batch of images (re-order channels).

    :param image:
    :return:
    """
    if len(image.shape) == 3:
        return np.transpose(image, (2, 0, 1))
    return np.transpose(image, (0, 3, 1, 2))


def maybe_transpose(observation: np.ndarray, observation_space: spaces.Space) -> np.ndarray:
    """
    Handle the different cases for images as PyTorch use channel first format.
    :param observation:
    :param observation_space:
    :return: channel first observation if observation is an image
    """
    if is_image_space(observation_space):
        if not (observation.shape == observation_space.shape or observation.shape[1:] == observation_space.shape):
            # Try to re-order the channels
            transpose_obs = transpose_image(observation)
            if transpose_obs.shape == observation_space.shape or transpose_obs.shape[1:] == observation_space.shape:
                observation = transpose_obs
