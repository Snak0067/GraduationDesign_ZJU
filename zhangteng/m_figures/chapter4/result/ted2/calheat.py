from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import os,json
import collections
from PIL import Image
import cv2
from scipy.spatial import distance


def get_pixel_distance(rgb_frame, fake_frame, total_distance, total_pixels, nmfc_frame=None):
    # If NMFC frame is given, use it as a mask.
    mask = None
    if nmfc_frame is not None:
        mask = np.sum(nmfc_frame, axis=2)
        mask = (mask > (np.ones_like(mask) * 0.01)).astype(np.int32)
    # Sum rgb distance across pixels.
    error = abs(rgb_frame.astype(np.int32) - fake_frame.astype(np.int32))
    if mask is not None:
        distance = np.multiply(np.linalg.norm(error, axis=2), mask)
        n_pixels = mask.sum()
    else:
        distance = np.linalg.norm(error, axis=2)
        n_pixels = distance.shape[0] * distance.shape[1]
    sum_distance = distance.sum()
    total_distance += sum_distance
    total_pixels += n_pixels
    # Heatmap
    maximum = 50.0
    minimum = 0.0
    maxim = maximum * np.ones_like(distance)
    distance_trunc = np.minimum(distance, maxim)
    zeros = np.zeros_like(distance)
    ratio = 2 * (distance_trunc-minimum) / (maximum - minimum)
    b = np.maximum(zeros, 255*(1 - ratio))
    r = np.maximum(zeros, 255*(ratio - 1))
    g = 255 - b - r
    heatmap = np.stack([r, g, b], axis=2).astype(np.uint8)
    if nmfc_frame is not None:
        heatmap = np.multiply(heatmap, np.expand_dims(mask, axis=2)).astype(np.uint8)
    # draw_str(heatmap, (20, 20), "%0.1f" % (sum_distance/n_pixels))
    return total_distance, total_pixels, heatmap