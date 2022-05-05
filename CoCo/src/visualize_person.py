from array import typecodes
import json
# from json.tool import main 
import os 
import time
# from matplotlib import image 
import numpy as np 
import math
import cv2 
from tqdm import tqdm
import random


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

def visualize_bbox(image, bbox,color):
    xtop, ytop, width, height = bbox
    image = cv2.rectangle(image, (int(xtop), int(ytop)),\
                            (int(xtop + width), int(ytop + height)), color,3)
    return image

def visualize_keypoint(image, keypoints, color):
    global join_keypoint_categories
    l_keypoint = []
    # print(len(keypoints))
    for i in range(17):
        x = int(keypoints[i * 3])
        y = int(keypoints[i * 3 + 1])
        l_keypoint.append((x,y))
        if x != 0 and y != 0:
            # print(x,y)
            image = cv2.circle(image, (x,y), 5,color, -1)
    for pair_point in join_keypoint_categories:
        start_point_index, end_point_index = pair_point
        if l_keypoint[start_point_index - 1][0] != 0 and l_keypoint[start_point_index - 1][1] != 0 and\
            l_keypoint[end_point_index - 1][0] != 0 and l_keypoint[end_point_index - 1][1] != 0:
            image = cv2.line(image, l_keypoint[start_point_index - 1], l_keypoint[end_point_index - 1], color, 2)
    return image 


def visualize_keypoint_cuong(image, keypoints, color):
    l_keypoint = []
    # print(len(keypoints))
    for i in range(15):
        x = int(keypoints[i * 3])
        y = int(keypoints[i * 3 + 1])
        l_keypoint.append((x,y))
        if x != 0 and y != 0:
            # print(x,y)
            image = cv2.circle(image, (x,y), 5,color, -1)
    return image 

def check_size_bbox(bbox):
    # config_threshold_small = 1024
    config_threshold_small = 2304
    xtop, ytop, width, height = bbox
    size_bbox = width * height 
    if size_bbox > config_threshold_small:
        return True 
    else: 
        return False

def check_size_bbox_ratio(bbox, size_image):
    xtop, ytop, width, height = bbox
    size_bbox = width * height 
    if size_bbox/size_image >= 0.005:
        return True
    else:
        return False


def valid_num_keypoint(l_annotation):
    for annotation in l_annotation:
        num_keypoints = annotation["num_keypoints"]
        if num_keypoints == 0:
            return True
    return False


if __name__ == '__main__':

    path_annotation = "/home/asilla/duongnh/datasets/coco_train.json"
    file_annotation = open(path_annotation,'r')
    json_data = json.load(file_annotation)
    l_image = json_data['images']
    l_annotations = json_data['annotations']
    l_categories = json_data['categories']
    global join_keypoint_categories
    join_keypoint_categories = l_categories[0]["skeleton"]

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
    l_color_keypointjoint = [(255,0,0), (0,255,255), (255,0,255),(0,140,255),(32,165,218),(0,128,128),(0,100,0),(79,79,47),(204,209,72),(112,25,25),(139,0,139)]
    count = 0
    # rs_folder = os.listdir("/home/asilla/duongnh/datasets/vis")

    count_vs = 0
    count_missing = 0

    #Customize
    path_dataset = "/home/asilla/duongnh/datasets/Filtered_data_29_03/Log_before_after"
    path_des_folder = "sample_before_after"
    # dataset = open(path_dataset, 'r')
    # l_item = []
    # for line in dataset:
    #     line = line[:-1]
    #     l_item.append(line)

    # random.shuffle(l_item)

    # num_sample = 2000
    
    for file in tqdm(os.listdir(path_dataset)):
        # if file in dict_after:
        #     continue
    
        if ".jpg" in file:
            image_id = dict_image_name[file]
            if image_id not in dict_image_id:
                count_missing += 1
                continue
            l_annotation = dict_image_id[image_id]
            # if not valid_num_keypoint(l_annotation):
            #     continue
            source_path_image = path_folder + "/" + file
            image = cv2.imread(path_folder + "/" + file)
            H,W,_ = image.shape
            # size_area_image = H * W
            status_write = False
            index_color_joint = 0
            for annotation in l_annotation:
                if annotation["iscrowd"] == 1:
                    continue
                num_keypoints = annotation["num_keypoints"]
                keypoint = annotation["keypoints"]
                bbox = annotation["bbox"]
                red = (0,0,255)
                blue_light  = (255,255,0)
                purple = (239,73,249)
                orange = (54, 183, 242)
                green = (30,255, 78)

                # type_bbox = check_size_bbox(bbox)
                color_join = l_color_keypointjoint[index_color_joint]
                if num_keypoints > 0:
                    image = visualize_keypoint_cuong(image, keypoint, color_join)
                    index_color_joint += 1
                    if index_color_joint > len(l_color_keypointjoint) - 1:
                        index_color_joint = 0
                    image = visualize_bbox(image, bbox, red)
                elif num_keypoints == 0 and check_size_bbox(bbox):
                    print("Found num keypoint = 0")
                    status_write = True
                    image = visualize_bbox(image, bbox, green)
            # if status_write:
            count_vs += 1
            des_path_image = path_des_folder + "/" + file
            cv2.imwrite(des_path_image, image)
    print(count_vs)
    print(count_missing)
    # print(count)
                    # print("Save {}".format(file))
                    
        #             count_image += 1
        # if count_image > 30:
        #     break