import numpy as np
import torch as th

red = [1, 0, 0]
green = [0, 1, 0]
blue = [0, 0, 1]


def get_all_green_pickup_obs():
    """
    Returns
    A JBW observation in which an agent is surrounded by nothing but green items.
    Stepping into any direction would lead to the collection of a green item.
    """
    pick_up_green = np.zeros((1 * 11 * 11 * 3), dtype=np.float).reshape(1, 11, 11, 3)
    pick_up_green[0, :, :, :] = green
    pick_up_green[0, 5, 5] = blue
    pick_up_green = np.moveaxis(pick_up_green, 3, 1)
    return th.tensor(pick_up_green, dtype=th.float)


def get_one_green_pickup_obs():
    """
    Returns
    A JBW observation in which an agent is surrounded by a cross of green items and otherwise white background.
    Stepping into any direction would lead to the collection of a green item.
    """
    pick_up_green = np.zeros((1 * 11 * 11 * 3), dtype=np.float).reshape(1, 11, 11, 3)
    pick_up_green[0, 4, 5, :] = green
    pick_up_green[0, 6, 5, :] = green
    pick_up_green[0, 5, 4, :] = green
    pick_up_green[0, 5, 6, :] = green
    pick_up_green[0, 5, 5] = blue
    pick_up_green = np.moveaxis(pick_up_green, 3, 1)
    return th.tensor(pick_up_green, dtype=th.float)


def get_all_red_pickup_obs():
    """
    Returns
    A JBW observation in which an agent is surrounded by nothing but red items.
    Stepping into any direction would lead to the collection of a red item.
    """
    pick_up_red = np.zeros((1 * 11 * 11 * 3), dtype=np.float).reshape(1, 11, 11, 3)
    pick_up_red[0, :, :, :] = red
    pick_up_red[0, 5, 5] = blue
    pick_up_red = np.moveaxis(pick_up_red, 3, 1)
    return th.tensor(pick_up_red, dtype=th.float)


def get_one_red_pickup_obs():
    """
    Returns
    A JBW observation in which an agent is surrounded by a cross of red items and otherwise white background.
    Stepping into any direction would lead to the collection of a red item.
    """
    pick_up_red = np.zeros((1 * 11 * 11 * 3), dtype=np.float).reshape(1, 11, 11, 3)
    pick_up_red[0, 4, 5, :] = red
    pick_up_red[0, 6, 5, :] = red
    pick_up_red[0, 5, 4, :] = red
    pick_up_red[0, 5, 6, :] = red
    pick_up_red[0, 5, 5] = blue
    pick_up_red = np.moveaxis(pick_up_red, 3, 1)
    return th.tensor(pick_up_red, dtype=th.float)


def get_white_obs():
    """
    Returns
    A JBW observation in which an agent is surrounded by nothing but white background pixels.
    """
    white_obs = np.zeros((1 * 11 * 11 * 3), dtype=np.float).reshape(1, 11, 11, 3)
    white_obs[0, 5, 5] = blue
    white_obs = np.moveaxis(white_obs, 3, 1)
    return th.tensor(white_obs, dtype=th.float)
