from utils import BoundBox, bbox_iou
from keras.utils import Sequence
import random
import math
import numpy as np
import pydicom
import os
from skimage.transform import resize

def parse_dataset(raw_annotations):
    images = {}

    for index, raw_annotation in raw_annotations.iterrows():
        current_bbox = BoundBox(raw_annotation.x,
                                raw_annotation.y,
                                raw_annotation.width,
                                raw_annotation.height,
                                raw_annotation.Target)

        if raw_annotation.patientId not in images:
            if current_bbox.label:
                images[raw_annotation.patientId] = [current_bbox]
        else:
            images[raw_annotation.patientId].append(current_bbox)

    return images


class BatchGenerator(Sequence):
    def __init__(self, images, config, shuffle=True):
        self.generator = None
        self.images = images
        self.config = config
        self.shuffle = shuffle
        self.anchors = [BoundBox(0, 0, config.anchors[2*i], config.anchors[2*i+1])
                        for i in range(int(len(config.anchors)//2))]

        self.image_keys = list(images.keys())

        if self.shuffle:
            random.shuffle(self.image_keys)

    def __len__(self):
        return math.ceil(len(self.images)/self.config.batch_size)

    def __getitem__(self, idx):
        scale = self.config.image_width/self.config.original_width
        l_bound = idx * self.config.batch_size
        r_bound = (idx + 1) * self.config.batch_size

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config.batch_size

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound,
                            self.config.image_height,
                            self.config.image_width,
                            3))
        b_batch = np.zeros((r_bound - l_bound,
                            1, 1, 1,
                            self.config.max_boxes_per_image, 4))
        y_batch = np.zeros((r_bound - l_bound,
                            self.config.grid_height,
                            self.config.grid_width,
                            5))

        for patient in self.image_keys[l_bound:r_bound]:
            image = self.load_image(patient)
            bboxes = self.images[patient]

            for box in bboxes:
                if box.xmax > box.xmin and box.ymax > box.ymin:
                    grid_cell_width = self.config.image_width/self.config.grid_width
                    grid_cell_height = self.config.image_height/self.config.grid_height

                    center_x = .5*(box.xmin + box.xmax)*scale
                    grid_x = int(np.floor(center_x / (float(self.config.image_width / self.config.grid_width))))
                    center_x = (center_x - grid_x * grid_cell_width) / grid_cell_width
                    center_y = .5 * (box.ymin + box.ymax)*scale
                    grid_y = int(np.floor(center_y / (float(self.config.image_height / self.config.grid_height))))
                    center_y = (center_y - grid_y * grid_cell_height) / grid_cell_height

                    if grid_x < self.config.grid_width and grid_y < self.config.grid_height:
                        center_w = (box.xmax*scale - box.xmin*scale) / self.config.image_width
                        center_h = (box.ymax*scale - box.ymin*scale) / self.config.image_height

                        box = [center_x, center_y, center_w, center_h]

                        y_batch[instance_count, grid_y, grid_x, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, 4] = 1.

            x_batch[instance_count] = image
            instance_count += 1

        return x_batch, y_batch

    def load_image(self, patientId):
        image_path = os.path.join(self.config.image_folder_path, patientId + '.dcm')
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array

        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

        return self.resize_and_norm_image(image)

    def resize_and_norm_image(self, image):
        return resize(image, (self.config.image_width, self.config.image_height),
               anti_aliasing=True)
