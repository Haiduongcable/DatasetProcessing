from array import typecodes
import json
from ntpath import join
# from json.tool import main 
import os 
import time
# from matplotlib import image 
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
                            (int(xtop + width), int(ytop + height)), color,1)
    return image

# def find_match_point(point1, l_join_point):
#     for pair_point in l_join_point:
#         start_point, end_point = pair_point
#         if start_point == point1:
def update_joint_kp(join_keypoint_categories):
    status_update = True
    update_joint = []
    for pair in join_keypoint_categories:
        if 0 in pair:
            status_update = False
            break 
    if status_update:
        for pair in join_keypoint_categories:
            start_index, end_index = pair 
            update_joint.append((start_index - 1, end_index - 1))
        return update_joint
    return join_keypoint_categories


def visualize_keypoint(image, keypoints, color):
    global join_keypoint_categories
    update_joint = update_joint_kp(join_keypoint_categories)
    l_keypoint = []
    for i in range(15):
        x = int(keypoints[i * 3])
        y = int(keypoints[i * 3 + 1])
        v = int(keypoints[i * 3 + 2])
        l_keypoint.append((x,y,v))
        if x != 0 and y != 0 and v != 0:
            # print(x,y)
            image = cv2.circle(image, (x,y), 2,color, -1)
    #print(join_keypoint_categories)
    for pair_point in update_joint:
        start_point_index, end_point_index = pair_point
        # print(start_point_index, end_point_index)
        # print(len(l_keypoint))
        if l_keypoint[start_point_index][2] != 0 and l_keypoint[end_point_index][2] != 0 :
            image = cv2.line(image, (l_keypoint[start_point_index][0], l_keypoint[start_point_index][1]),\
                        (l_keypoint[end_point_index][0], l_keypoint[end_point_index][1]), color, 1)
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
    elif config_threshold_medium_large <= size_bbox <= config_threshold_large:
        return "medium_large"
    else:
        return "large"


def valid_num_keypoint(l_annotation):
    for annotation in l_annotation:
        num_keypoints = annotation["num_keypoints"]
        if num_keypoints == 0:
            return True
    return False


if __name__ == '__main__':

    path_annotation = "/home/asilla/duongnh/datasets/CrownPose/ConvertAnnotation/Converted_Annotation_train_hrnet.json"
   
    #path_annotation = "/home/asilla/duongnh/datasets/annotations/annotations/person_keypoints_val2017.json"
    file_annotation = open(path_annotation,'r')
    json_data = json.load(file_annotation)
    l_image = json_data['images']
    l_annotations = json_data['annotations']
    l_categories = json_data['categories']
    global join_keypoint_categories
    join_keypoint_categories = l_categories[0]["skeleton"]
    print(join_keypoint_categories)
    
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color = (0, 0, 255)
    thickness = 2
    l_color_keypointjoint = [(255,0,0), (0,255,255), (255,0,255),(0,140,255),(32,165,218),(0,128,128),(0,100,0),(79,79,47),(204,209,72),(112,25,25),(139,0,139)]
    count = 0
    path_folder_source = "/home/asilla/duongnh/datasets/CrownPose/images"
    count_visualize = 0
    # print(dict_image_name.keys())
    for file in sorted(os.listdir(path_folder_source)):
        if file in dict_image_name:
            image_id = dict_image_name[file]
            l_annotation = dict_image_id[image_id]
            
            image = cv2.imread(path_folder_source + "/" + file)
            index_color_joint = 0
            for annotation in l_annotation:
                num_keypoints = annotation["num_keypoints"]
                keypoint = annotation["keypoints"]
                bbox = annotation["bbox"]
                red = (0,0,255)
                blue_light  = (255,255,0)
                purple = (239,73,249)
                orange = (54, 183, 242)
                green = (30,255, 78)

                type_bbox = check_size_bbox(bbox)
                color_join = l_color_keypointjoint[index_color_joint]
                if num_keypoints > 0:
                    image = visualize_bbox(image, bbox, red)
                    image = visualize_keypoint(image, keypoint, color_join)
                    index_color_joint += 1
                    if index_color_joint > len(l_color_keypointjoint) - 1:
                        index_color_joint = 0
            cv2.imwrite("/home/asilla/duongnh/project/Analys_CrownPose/visualize/train/" + file, image)
            count_visualize += 1
            if count_visualize > 100:
                break
