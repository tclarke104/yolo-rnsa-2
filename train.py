import os
import pandas as pd
from utils import Config
import preprocessing
import cv2
import numpy as np
import pydicom
import model as yolo
from keras.optimizers import Adam

# Declare file path constants
KAGGLE_DATA_ROOT = os.path.abspath('/Users/travisclarke/kaggle-data')
TRAIN_DIR = os.path.join(KAGGLE_DATA_ROOT, 'stage_1_train_images')
TEST_DIR = os.path.join(KAGGLE_DATA_ROOT, 'stage_1_test_images')
ANNOTATIONS_FILE = os.path.join(KAGGLE_DATA_ROOT, 'stage_1_train_labels.csv')

# load and parse images and annotations
raw_annotations = pd.read_csv(ANNOTATIONS_FILE)
images = preprocessing.parse_dataset(raw_annotations)

config = Config(image_height=448,
                image_width=448,
                grid_height=13,
                grid_width=13,
                num_boxes=5,
                labels=['Pneumonia'],
                num_labels=1,
                anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                batch_size=5,
                max_boxes_per_image=50,
                image_folder_path=TRAIN_DIR,
                original_height=1024,
                original_width=1024)

generator = preprocessing.BatchGenerator(images,config, False)
batch_x, batch_y = generator[0]

model = yolo.build_tiny_model()
optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss=yolo.custom_loss_function, optimizer=optimizer)
model.fit_generator(generator, steps_per_epoch=len(generator), epochs=10)
# def get_coord_of_pos(mat):
#     for i in range(0,13):
#         for j in range(0,13):
#             for x in range(0,5):
#                 if mat[i,j,x,4] == 1:
#                     return i,j,x
#     return None
#
#
# image_names = list(images.keys())
# annotation = images[image_names[0]][0]
# print(images[image_names[0]])
# img_path = os.path.join(TRAIN_DIR, image_names[0] + '.dcm')
# ds = pydicom.dcmread(img_path)
# image = ds.pixel_array
# xmin = int(annotation.xmin)
# xmax = int(annotation.xmax)
# ymin = int(annotation.ymin)
# ymax = int(annotation.ymax)
# center_x_1 = int((xmin + xmax)/2)
# center_y_1 = int((ymin + ymax)/2)
# cv2.rectangle(image, (xmin,ymin), (int(annotation.xmax), int(annotation.ymax)), 0, 5)
# cv2.circle(image,(center_x_1,center_y_1), 5, 0, -1)
# cv2.imshow('truth', image)
#
#
# for idx, img in enumerate(batch_x):
#
#     if np.max(batch_y[idx]) > 0:
#         grid_cell_width = 416/13
#         grid_cell_height = 416/13
#
#         grid_x, grid_y, anchor = get_coord_of_pos(batch_y[idx])
#         print(grid_x)
#         print(grid_y)
#         print(anchor)
#         label = batch_y[idx, grid_x,grid_y, anchor]
#         width = label[2] * 416
#         height = label[3] * 416
#         xmin = int((grid_cell_width * label[0]) + grid_cell_width * grid_x - width*.5)
#         ymin = int((grid_cell_height * label[1]) + grid_cell_height * grid_y - height*.5)
#         xmax = int((grid_cell_width * label[0]) + grid_cell_width * grid_x + width*.5)
#         ymax = int((grid_cell_height * label[1]) + grid_cell_height * grid_y + height*.5)
#         # center_x = int((416 / 13) * (grid_x+1) + label[0])
#         # center_y = int((416 / 13) * (grid_y+1) + label[1])
#         # cv2.circle(img, (center_x, center_y), 5, 0, -1)
#         cv2.rectangle(img, (xmin, ymin), (xmax,ymax),
#                       0, 5)
#     cv2.imshow('image', img)
#     cv2.waitKey(0)