import os
import glob
import cv2
import numpy
import xml.etree.ElementTree as ET
import tqdm

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 
               'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 
               'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 
               'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

classes = list(classes_num.keys())


def voc_load_data(img_dir_path, annotation_path, batch=10):
    images, labels = [], []
    img_file_list = glob.glob((img_dir_path + "/*.jpg"))

    for i in range(len(img_file_list)):
        for img_path in tqdm.tqdm(img_file_list[batch * i: batch * (i + 1)]):

            # Read image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_h, image_w = image.shape[0:2]

            # Resize & normalization
            image = cv2.resize(image, (448, 448))
            image = image / 255.0

            images.append(image)

            # Read xml file
            xml_name = os.path.split(img_path)[-1]
            xml_name = xml_name.split(".")[-2]
            xml_path = annotation_path + f"/{xml_name}.xml"

            # parse xml
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Empty matrix
            label_matrix = numpy.zeros((7, 7, 25))

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                class_name = obj.find('name').text
                if class_name not in classes or difficult == "1":
                    continue

                # Set class id
                cls_id = classes.index(class_name)
                xmlbox = obj.find('bndbox')
                tlx, tly = int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text)
                brx, bry = int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)

                # point -> 0~1 normalization
                x = (tlx + brx) / 2 / image_w
                y = (tly + bry) / 2 / image_h
                w = (brx - tlx) / image_w
                h = (bry - tly) / image_h

                # loc in 7x7 grid & point(0~1) in grid cell
                loc = [7 * x, 7 * y]
                loc_i = int(loc[1])
                loc_j = int(loc[0])
                y = loc[1] - loc_i
                x = loc[0] - loc_j

                if label_matrix[loc_i, loc_j, 24] == 0:
                    # [<----------20---------->|x|y|w|h|pc]
                    label_matrix[loc_i, loc_j, cls_id] = 1
                    label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
                    label_matrix[loc_i, loc_j, 24] = 1  # response

            labels.append(label_matrix)

        return numpy.array(images), numpy.array(labels)