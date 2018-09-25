from collections import namedtuple


class BoundBox(object):
    def __init__(self, x_min, y_min, width, height, target=0):
        self.xmin = x_min
        self.ymin = y_min
        self.xmax = x_min + width
        self.ymax = y_min + height
        self.label = target


Config = namedtuple('Config',
                    ['image_height',
                     'image_width',
                     'grid_height',
                     'grid_width',
                     'num_boxes',
                     'labels',
                     'num_labels',
                     'anchors',
                     'batch_size',
                     'max_boxes_per_image',
                     'image_folder_path',
                     'original_height',
                     'original_width']
                    )


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3
