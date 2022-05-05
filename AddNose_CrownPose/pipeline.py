
import numpy as np 
import cv2 
import os 
import time 
from retinaface_detection import build_model, detection_face
import json 
from tqdm import tqdm 
from inference_hrnet import HumanPose_Estimation

def visualize_keypoint(image, nose_kp):
    image = cv2.circle(image, (int(nose_kp[0]), int(nose_kp[1])), 2, (0, 255, 255), 2)
    return image


    
def interpolate_nose_keypoint(l_kp):
    '''
    '''
    head_kp = (int(l_kp[12 * 3]), int(l_kp[12 * 3 + 1]))
    neck_kp = (int(l_kp[13 * 3]), int(l_kp[13 * 3 + 1]))
    
    nose_kp = (int((head_kp[0] * 2 + neck_kp[0] * 3) / 5), 
               int((head_kp[1] * 2 + neck_kp[1] * 3) / 5))
    return nose_kp



def not_have_head_neck(l_kp):
    '''
    12 -> 13
    '''
    
    head_kp = (int(l_kp[12 * 3]), int(l_kp[12 * 3 + 1]))
    neck_kp = (int(l_kp[13 * 3]), int(l_kp[13 * 3 + 1]))
    if (head_kp[0] == 0 and head_kp[1] == 0)  or (neck_kp[0] == 0 and neck_kp[1] == 0):
        return True 
    return False

def add_keypoint_annotation(image, l_annotation, resnet_model):
    '''
    Args: image np.array: source image 
          annotation: dict: annotation per image
          index_source: index of annotation in source annotation format
    Returns: Add 1 keypoint to annotation and reupdate in source annotation
    Config: Head: 12th, Neck: 13th keypoint ( start from 0)
    Flow: 
    Loop all annotation in annotations per image 
            if num_kp > 0 and having head keypoint:
                crop image -> retinaface model 
            if have result -> add to annotation and skeleton 
            else: interpolate nose keypoint from head keypoint and neck keypoint 
        return  
    '''
    image_draw = image.copy()
    H, W,_ = image.shape
    l_update_annotation = []
    for (index_source, annotation) in l_annotation:
        num_keypoints = annotation["num_keypoints"]
        keypoints = annotation["keypoints"]
        bbox = annotation["bbox"]
        if num_keypoints == 0 or not_have_head_neck(keypoints):
            continue
        xtop, ytop, width, height = bbox
        xtop = max(int(xtop),0)
        ytop = max(int(ytop),0)
        xbottom = min(int(xtop + width), W)
        ybottom = min(int(ytop + height),H)
        image_crop = image[ytop:ybottom, xtop:xbottom,:]
        nose_coordinate_pred, status_detection = detection_face(resnet_model, image_crop)
        if not status_detection: 
            #nose_coordinate = interpolate_nose_keypoint(keypoints)
            continue
        else:
            nose_coordinate = (nose_coordinate_pred[0] + int(bbox[0]), nose_coordinate_pred[1] + int(bbox[1]))
            keypoints = keypoints + [nose_coordinate[0], nose_coordinate[1], 1]
            annotation["num_keypoints"] += 1
            annotation["keypoints"] = keypoints
            annotation["bbox"] = bbox 
            l_update_annotation.append((index_source, annotation))
    return l_update_annotation
        
        
def add_keypoint_hrnet(image, l_annotation, hrnet):
    global count_detection_nose 
    global total_nose
    '''
    Args: image np.array: source image 
          annotation: dict: annotation per image
          index_source: index of annotation in source annotation format
    Returns: Add 1 keypoint to annotation and reupdate in source annotation
    Config: Head: 12th, Neck: 13th keypoint ( start from 0)
    Flow: 
    Loop all annotation in annotations per image 
            if num_kp > 0 and having head keypoint:
                crop image -> retinaface model 
            if have result -> add to annotation and skeleton 
            else: interpolate nose keypoint from head keypoint and neck keypoint 
        return  
    '''
    image_draw = image.copy()
    H, W,_ = image.shape
    l_update_annotation = []
    for (index_source, annotation) in l_annotation:
        num_keypoints = annotation["num_keypoints"]
        keypoints = annotation["keypoints"]
        bbox = annotation["bbox"]
        iscrowd = int(annotation["iscrowd"])
        if iscrowd == 1:
            continue
        if num_keypoints == 0 or not_have_head_neck(keypoints):
            continue
        xtop, ytop, width, height = bbox
        xtop = max(int(xtop),0)
        ytop = max(int(ytop),0)
        xbottom = min(int(xtop + width), W)
        ybottom = min(int(ytop + height),H)
        image_crop = image[ytop:ybottom, xtop:xbottom,:]
        nose_coordinate_pred, status_detection, confidence_score = hrnet.predict(image_crop)
        if not status_detection: 
            #nose_coordinate = interpolate_nose_keypoint(keypoints)
            continue
        else:
            nose_coordinate = (nose_coordinate_pred[0] + int(bbox[0]), nose_coordinate_pred[1] + int(bbox[1]))
            total_nose += 1
            if confidence_score > 0.75:
                keypoints = keypoints + [nose_coordinate[0], nose_coordinate[1], 1]
                annotation["num_keypoints"] += 1
                annotation["keypoints"] = keypoints
                annotation["bbox"] = bbox
                annotation["confidence_score_nose"] = str(confidence_score)
                count_detection_nose += 1
                l_update_annotation.append((index_source, annotation))
            
    return l_update_annotation
    
    
    
if __name__ == '__main__':
    global count_detection_nose 
    global total_nose 
    total_nose = 0
    count_detection_nose = 0
    path_annotation = "/home/asilla/duongnh/datasets/CrownPose/CrowdPose/crowdpose_test.json"
    path_folder_source = "/home/asilla/duongnh/datasets/CrownPose/images"
    file_annotation = open(path_annotation, "r")
    dataset = json.load(file_annotation)
    log_dataset = dataset.copy()
    info_dataset = dataset["info"]
    categories_dataset = dataset["categories"]
    images_dataset = dataset["images"]
    annotations_dataset = dataset["annotations"]
    hrnet = HumanPose_Estimation()
    
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
        image = cv2.imread(path_folder_source + "/" + file)
        image_id = dict_image_name[file]
        l_annotation = dict_image_id[image_id]
        l_update_annotation = add_keypoint_hrnet(image, l_annotation, hrnet)
        
        for (index_source, update_annotation) in l_update_annotation:
            annotations_dataset[index_source] = update_annotation
        
    log_dataset["annotations"] = annotations_dataset
    json_str = json.dumps(log_dataset)
    with open("Annotation_test_hrnet.json", "w") as f:
        f.write(json_str)
    print(total_nose)
    print(count_detection_nose)