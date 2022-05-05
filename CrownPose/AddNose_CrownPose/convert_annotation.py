
import numpy as np 
import cv2 
import os 
import time 
import json 
from tqdm import tqdm 


def visualize_keypoint(image, nose_kp):
    image = cv2.circle(image, (int(nose_kp[0]), int(nose_kp[1])), 2, (0, 255, 255), 2)
    return image


    
# def interpolate_nose_keypoint(l_kp):
#     '''
#     '''
#     head_kp = (int(l_kp[12 * 3]), int(l_kp[12 * 3 + 1]))
#     neck_kp = (int(l_kp[13 * 3]), int(l_kp[13 * 3 + 1]))
    
#     nose_kp = (int((head_kp[0] * 2 + neck_kp[0] * 3) / 5), 
#                int((head_kp[1] * 2 + neck_kp[1] * 3) / 5))
#     return nose_kp



# def not_have_head_neck(l_kp):
#     '''
#     12 -> 13
#     '''
    
#     head_kp = (int(l_kp[12 * 3]), int(l_kp[12 * 3 + 1]))
#     neck_kp = (int(l_kp[13 * 3]), int(l_kp[13 * 3 + 1]))
#     if (head_kp[0] == 0 and head_kp[1] == 0)  or (neck_kp[0] == 0 and neck_kp[1] == 0):
#         return True 
#     return False

def convert_keypoint(l_keypoint, log_index):
    l_cv_keypoint = []
    for tmp_index in log_index:
        x_kp = l_keypoint[tmp_index * 3]
        y_kp = l_keypoint[tmp_index * 3 + 1]
        visible = l_keypoint[tmp_index * 3 + 2]
        l_cv_keypoint.append(x_kp)
        l_cv_keypoint.append(y_kp)
        l_cv_keypoint.append(visible)
    return l_cv_keypoint

def update_neck_keypoint(l_keypoint):
    '''
    Update Neck keypoint base on left shoulder and right shoulder 
    Args: l_keypoint : list of keypoint (3 * 14)
    Return l_update_neck_keypoint
    '''
    l_update_neck_kp = l_keypoint.copy()
    index_neck_kp = 1
    index_left_shoulder = 5 
    index_right_shoulder = 2 
    x_left_shoulder, y_left_shoulder = l_keypoint[index_left_shoulder * 3], l_keypoint[index_left_shoulder * 3 + 1]
    left_shoulder_visible = l_keypoint[index_left_shoulder * 3 + 2]
    
    x_right_shoulder, y_right_shoulder = l_keypoint[index_right_shoulder * 3], l_keypoint[index_right_shoulder * 3 + 1]
    right_shoulder_visible = l_keypoint[index_right_shoulder * 3 + 2]
    
    x_neck = int((x_left_shoulder + x_right_shoulder)/2)
    y_neck  = int((y_left_shoulder + y_right_shoulder)/2)
    if left_shoulder_visible == 0 or right_shoulder_visible == 0:
        neck_visible = 0 
    elif left_shoulder_visible == 1 or right_shoulder_visible == 1:
        neck_visible = 1 
    else:
        neck_visible = 2
    l_update_neck_kp[index_neck_kp * 3] = x_neck
    l_update_neck_kp[index_neck_kp * 3 + 1]= y_neck 
    l_update_neck_kp[index_neck_kp * 3 + 2] = neck_visible
    return l_update_neck_kp

def append_centeroid(l_kp):
    '''
    
    '''
    index_LS = 5 
    index_RS = 2 
    index_LH = 11
    index_RH = 8
    x_LH, y_LH, v_LH = l_kp[index_LH * 3],  l_kp[index_LH * 3 + 1],  l_kp[index_LH * 3 + 2]
    x_RS, y_RS, v_RS = l_kp[index_RS * 3], l_kp[index_RS * 3 + 1], l_kp[index_RS * 3 + 2]
    x_LS, y_LS, v_LS = l_kp[index_LS * 3], l_kp[index_LS * 3 + 1], l_kp[index_LS * 3 + 2]
    x_RH, y_RH, v_RH = l_kp[index_RH * 3], l_kp[index_RH * 3 + 1], l_kp[index_RH * 3 + 2]
    average = False
    #Note object 
    
    # interpolate
    if ((v_LS == 2) and (v_RH == 2)):
        if (v_RS == 2 and v_LH == 2):
            average = True
        x_s = x_LS
        y_s = y_LS
        x_h = x_RH
        y_h = y_RH
        v_center = 2
    elif (v_RS == 2 and v_LH == 2):
        x_s = x_RS
        y_s = y_RS 
        x_h = x_LH
        y_h = y_LH 
        v_center = 2
    elif ((v_LS != 0) and (v_RH != 0)):
        if ((v_RS != 0) and (v_LH != 0)):
            average = True
        x_s = x_LS 
        y_s = y_LS 
        x_h = x_RH
        y_h = y_RH
        
        v_center = 1
    elif ((v_RS != 0) and (v_LH != 0)):
        x_s = x_RS
        y_s = y_RS
        x_h = x_LH
        y_h = y_LH
        
        v_center = 1
    else:
        x_s = x_RS
        y_s = y_RS
        x_h = x_LH
        y_h = y_LH
        v_center = 0
    if average == True:
        x_center = round((x_LS + x_RS + x_LH + x_RH) / 4.0)
        y_center = round((y_LS + y_RS + y_LH + y_RH) / 4.0)
    else:
        x_center = round((x_s + x_h) / 2.0)
        y_center = round((y_s + y_h) / 2.0)
    l_kp += [x_center, y_center, v_center]
    return l_kp



def cv_annotation(l_annotation):
    l_update_annotation = []
    log_index = [14,13,1,3,5,0,2,4,7,9,11,6,8,10]
    for (index_source, annotation) in l_annotation:
        cv_annotation = annotation.copy()
        num_keypoints = annotation["num_keypoints"]
        keypoints = annotation["keypoints"]
        if len(keypoints) == 42:
            keypoints += [0,0,0]
        bbox = annotation["bbox"]
        cv_keypoints = convert_keypoint(keypoints, log_index)
        annotation["num_keypoints"] = num_keypoints + 1
        update_kps = update_neck_keypoint(cv_keypoints)
        update_centroid = append_centeroid(update_kps)
        cv_annotation["keypoints"] = update_centroid
        
        l_update_annotation.append((index_source, cv_annotation))
    return l_update_annotation
    
if __name__ == '__main__':
    log_categories = {"supercategory": "person", "id": 1, "name": "person", "keypoints": ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "centeroid"], "skeleton": [[2, 1], [2, 3], [2, 6], [2, 15], [15, 3], [15, 6], [15, 9], [15, 12], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 14]]}
    path_annotation = "Annotation_test_hrnet.json"
    path_folder_source = "/home/asilla/duongnh/datasets/CrownPose/images"
    file_annotation = open(path_annotation, "r")
    dataset = json.load(file_annotation)
    log_dataset = dataset.copy()
    info_dataset = dataset["info"]
    categories_dataset = dataset["categories"]
    
    images_dataset = dataset["images"]
    annotations_dataset = dataset["annotations"]
    
    dict_image_name = {}
    dict_image_id = {}

    for info_image in images_dataset:
        image_id = info_image["id"]
        image_name = info_image["file_name"]
        dict_image_name[image_name] = image_id

    for index, annotation in enumerate(annotations_dataset):
        image_id = annotation["image_id"]
        num_keypoints = annotation["num_keypoints"]
        keypoint = annotation["keypoints"]
        bbox = annotation["bbox"]
        
        if image_id not in dict_image_id:
            dict_image_id[image_id] = []
            dict_image_id[image_id].append((index, annotation))
        else:
            dict_image_id[image_id].append((index, annotation))
            
    for file in tqdm(sorted(os.listdir(path_folder_source))):
        if file not in dict_image_name:
            continue 
        image_id = dict_image_name[file]
        l_annotation = dict_image_id[image_id]
        l_update_annotation = cv_annotation(l_annotation)
        
        for (index_source, update_annotation) in l_update_annotation:
            annotations_dataset[index_source] = update_annotation
        
    log_dataset["annotations"] = annotations_dataset
    log_categories = [{"supercategory": "person", "id": 1, "name": "person", "keypoints": ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "centeroid"], "skeleton": [[2, 1], [2, 3], [2, 6], [2, 15], [15, 3], [15, 6], [15, 9], [15, 12], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 14]]}]
    log_dataset["categories"] = log_categories
    json_str = json.dumps(log_dataset)
    with open("Converted_Annotation_test_hrnet.json", "w") as f:
        f.write(json_str)