import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import cv2 as cv
import xml.etree.ElementTree as ET
from google.colab import drive
from google.colab.patches import cv2_imshow

drive.mount('/content/drive/')

DATADIR_ANNOTATIONS = '/content/drive/MyDrive/Datasets/People_with_and_without_mask/annotations/'
DATADIR_IMAGES = '/content/drive/MyDrive/Datasets/People_with_and_without_mask/images/'
label2category = {'without_mask':0, 'mask_weared_incorrect':1, 'with_mask':2}
datas = []

def create_training_data():
    for xml_file in os.listdir(DATADIR_ANNOTATIONS):
        root_node = ET.parse(os.path.join(DATADIR_ANNOTATIONS, xml_file))
        data = {'path': None, 'objects': []}
        data['path'] = os.path.join(DATADIR_IMAGES, root_node.find('filename').text)
        for tag in root_node.findall('object'):
            label = label2category[tag.find('name').text]
            xmin = int(tag.find('bndbox/xmin').text)
            ymin = int(tag.find('bndbox/ymin').text)
            xmax = int(tag.find('bndbox/xmax').text)
            ymax = int(tag.find('bndbox/ymax').text)
            data['objects'].append([label, xmin, ymin, xmax, ymax])
        datas.append(data)

create_training_data()

for i in range(10):
    index = np.random.randint(0, len(datas)-1)
    img = cv.imread(datas[index]['path'], cv.IMREAD_COLOR)

    for (label, xmin, ymin, xmax, ymax) in datas[index]['objects']:
        if label == 0:
            cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
        elif label == 1:
            cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,255), 1)
        else:
            cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 1)

    cv2_imshow(img)

for data in datas:
  img = cv.imread(data['path'])
  for (category, xmin, ymin, xmax, ymax) in data['objects']:
    roi = img[ymin:ymax, xmin:xmax]
    roi = cv.resize(roi, (IMG_SIZE, IMG_SIZE))
    features = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
    label = keras.utils.to_categorical(category, 3)
    x.append(features)
    y.append(label)

x = np.array(x)
y = np.array(y)
