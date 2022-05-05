from array import typecodes
import json
import os 
import time
import numpy as np 
import math
import cv2 



def load_annotation(path_annotation):
    file_annotation = open(path_annotation,'r')
    json_data = json.load(file_annotation)
    l_image = json_data['images']
    l_annotations = json_data['annotations']
    l_categories = json_data['categories']
    l_bbox = []
    l_keypoint = []
    l_num_keypoint = []
    for annotation in l_annotations:
        image_id = annotation["image_id"]
        num_keypoints = annotation["num_keypoints"]
        keypoint = annotation["keypoints"]
        bbox = annotation["bbox"]
        l_bbox.append(bbox)
        l_num_keypoint.append(num_keypoints)
        l_keypoint.append(keypoint)
    return l_bbox, l_keypoint, l_num_keypoint




def visualize(path_annotation):
    l_bbox, l_keypoint, l_num_keypoint = load_annotation(path_annotation)



def visualize_bbox(image, bbox,color):
    xtop, ytop, width, height = bbox
    image = cv2.rectangle(image, (int(xtop), int(ytop)),\
                            (int(xtop + width), int(ytop + height)), color,3)
    return image
def visualize_keypoint(image, keypoints):
    for i in range(17):
        x = int(keypoints[i * 3])
        y = int(keypoints[i * 3 + 1])
        if x != 0 and y != 0:
            # print(x,y)
            image = cv2.circle(image, (x,y), 2,(0,255,0), -1)
    return image 

def check_size_bbox(bbox):
    config_threshold_small = 1024
    config_threshold_medium_large = 64 * 64
    config_threshold_large = 9216
    xtop, ytop, width, height = bbox
    size_bbox = width * height 
    if size_bbox < config_threshold_small:
        return "small"
    elif config_threshold_small <= size_bbox < config_threshold_medium_large:
        return "small_medium"
    elif config_threshold_medium_large <= config_threshold_large:
        return "medium_large"
    else:
        return "large"
if __name__ == '__main__':

    path_annotation = "/home/asilla/duongnh/datasets/annotations/person_keypoints_train2017_old.json"
    file_annotation = open(path_annotation,'r')
    json_data = json.load(file_annotation)
    l_image = json_data['images']
    l_annotations = json_data['annotations']
    l_categories = json_data['categories']
    dict_image_name = {}
    dict_image_id = {}

    for info_image in l_image:
        image_id = info_image["id"]
        image_name = info_image["file_name"]
        dict_image_name[image_name] = image_id
    
    for annotation in l_annotations:
        
        image_id = annotation["image_id"]
        num_keypoints = annotation["num_keypoints"]
        keypoint = annotation["keypoints"]
        bbox = annotation["bbox"]
        
        if image_id not in dict_image_id:
            dict_image_id[image_id] = []
            dict_image_id[image_id].append(annotation)
        else:
            dict_image_id[image_id].append(annotation)
    
    path_folder = "/home/asilla/duongnh/datasets/train2017"
    count_image = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color = (0, 0, 255)
    thickness = 2

    for file in os.listdir(path_folder):
        if ".jpg" in file:
            
            image_id = dict_image_name[file]
            if image_id not in dict_image_id:
                # print("not found: ", file)
                continue
            l_annotation = dict_image_id[image_id]
            tmp_log = False
            for annotation in l_annotation:
                if annotation["num_keypoints"] == 0:
                    tmp_log = True 
                    break

            if tmp_log:
                image = cv2.imread(path_folder + "/" + file)
                H,W,_ = image.shape
                for annotation in l_annotation:
                    if annotation["iscrowd"]:
                        continue
                    num_keypoints = annotation["num_keypoints"]
                    keypoint = annotation["keypoints"]
                    bbox = annotation["bbox"]
                    red = (0,0,255)
                    blue_light  = (255,255,0)
                    purple = (239,73,249)
                    orange = (54, 183, 242)
                    green = (30,255, 78)

                    type_bbox = check_size_bbox(bbox)
                    count = 0
                    #H: 384, W:288 ratio H/W: 4/3
                    if num_keypoints == 0 and (type_bbox == "medium_large" or type_bbox == "large"):
                        count += 1
                        x, y, w,h = bbox
                        size_crop = max(w,h)
                        # print(w,h)
                        # padding_size = size_crop * 1.2
                        # x_center = x + w/2
                        # y_center = y + h/2
                        # x_top_crop = max(int(x_center - padding_size/2),0)
                        # y_top_crop = max(int(y_center - padding_size/2), 0)
                        
                        # x_bottom_crop = min(int(x_center + padding_size/2), W)
                        # y_bottom_crop = min(int(y_center + padding_size/2), H)
                        x1 = int(x)
                        y1 = int(y)
                        x2 = int(x + w)
                        y2 = int(y + h)
                        
                        correction_factor = 384 / 288 * w / h
                        if correction_factor > 1:
                            # increase y side
                            center = y1 + (y2 - y1) // 2
                            length = int(round((y2 - y1) * correction_factor))
                            y1 = max(0, center - length // 2)
                            y2 = min(image.shape[0], center + length // 2)
                        elif correction_factor < 1:
                            # increase x side
                            center = x1 + (x2 - x1) // 2
                            length = int(round((x2 - x1) * 1 / correction_factor))
                            x1 = max(0, center - length // 2)
                            x2 = min(image.shape[1], center + length // 2)



                        if x1 == 0 or y1 == 0 or x2 == W or y2 == H:
                            continue
                        image_crop = image[y1:y2, x1:x2,:]
                        # image_crop = image[int(y):int(y + h), int(x):int(x + w),:]
                        name_image = file[:-4] + "_" + str(count) + ".jpg"
                        cv2.imwrite("crop_image/" + name_image, image_crop)
        #         count_image += 1
        # if count_image > 30:
        #     break