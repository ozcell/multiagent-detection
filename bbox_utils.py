## Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
import numpy as np
import torch as K

def relative_to_point(boxes, dtype='tensor'):
    """ Convert (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) (xmin, ymin, w, h) formed bboxes
    Return:
        boxes: (tensor) (xmin, ymin, xmax, ymax) formed bboxes
    """
    if dtype == 'tensor':
        return K.cat((boxes[:, :2],                    # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:]), 1) # xmax, ymax
    elif dtype == 'numpy':
        return np.concatenate((boxes[:, :2],                    # xmin, ymin
                               boxes[:, :2] + boxes[:, 2:]), 1) # xmax, ymax 


def point_to_relative(boxes):
    """ Convert (xmin, ymin, xmax, ymax) to (xmin, ymin, w, h)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) (xmin, ymin, xmax, ymax) formed bboxes
    Return:
        boxes: (tensor) (xmin, ymin, w, h) formed bboxes
    """
    return K.cat((boxes[:, :2],                    # xmin, ymin
                  boxes[:, 2:] - boxes[:, :2]), 1) # xmax, ymax


def intersect(box_a, box_b):
    # Expects (xmin, ymin, xmax, ymax) formed bboxes
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    
    max_xy = K.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                   box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    
    min_xy = K.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                   box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    
    inter = K.clamp((max_xy - min_xy), min=0)
    
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    # Expects (xmin, ymin, xmax, ymax) formed bboxes
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    
    inter = intersect(box_a, box_b)
    
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    
    union = area_a + area_b - inter
    
    return inter / union # [A,B]