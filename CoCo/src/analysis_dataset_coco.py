import os
import json 
import time 
import cv2 
import numpy as np
from pip import main


#Analysis Person, Motorbike annotation


def extract_person_overlap_motor_bike(l_annotations):
    '''
    Loop annotation
    
    count person 
    count_motorbike
    create dict image id with {person_bbox: [], motorbike_bbox: []}
    loop all key 
    if person > 0, motorbike > 0: 
    loop person 
        loop mortorbike: -> check IOU: 50% -> l_image_id append -> count += 1
    '''
    dict_image_id = {}
    count_person_bbox = 0
    count_motor_bike = 0
    object_id_person = 1
    object_id_motorbike = 4
    object_id_cat = 16
    object_id_dog = 17
    for annotation in l_annotations:
        object_id = annotation['category_id']
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        # keypoint_object = annotation['']
        if object_id == object_id_motorbike:
            count_motor_bike += 1
            if image_id not in dict_image_id:
                dict_image_id[image_id] = {}
                dict_image_id[image_id]['person'] = []
                dict_image_id[image_id]['mortorbike'] = []
                dict_image_id[image_id]['mortorbike'].append(bbox)
            else:
                dict_image_id[image_id]['mortorbike'].append(bbox)
        elif object_id == object_id_person:
            count_person_bbox += 1
            if image_id not in dict_image_id:
                dict_image_id[image_id] = {}
                dict_image_id[image_id]['person'] = []
                dict_image_id[image_id]['mortorbike'] = []
                dict_image_id[image_id]['person'].append(bbox)
            else:
                dict_image_id[image_id]['person'].append(bbox)
    return dict_image_id
            
    
def load_coco_annotation(path_annotation):
    '''
    Args: path_annotation: path to instance json file 
    '''
    file_annotation = open(path_annotation,'r')
    json_data = json.load(file_annotation)
    l_info = json_data['info']
    l_image = json_data['images']
    l_annotations = json_data['annotations']
    l_categories = json_data['categories']

    imageid2imagename = {}
    for info_image in l_image:
        file_name = info_image["file_name"]
        image_id = info_image["id"]
        imageid2imagename[image_id] = file_name
    return l_annotations, imageid2imagename


if __name__ == '__main__':
    path_npy = "/home/asilla/duongnh/project/Analys_COCO/motorbike_image_name.npy"
    l_motorbike_npy = np.load(path_npy)
    print(l_motorbike_npy)
    
    
    #path to instance coco json annotation
    path_annotation = "/home/asilla/duongnh/datasets/instances_train2017.json"
    l_annotations, imageid2imagename = load_coco_annotation(path_annotation)
    dict_image_id_motorbike = extract_person_overlap_motor_bike(l_annotations)

    l_image_have_motorbike = []

    l_image_id = []
    for image_id in dict_image_id_motorbike.keys():
        item_tmp = dict_image_id_motorbike[image_id]
        l_bbox_motorbike =item_tmp['mortorbike']
        l_bbox_person = item_tmp['person']
        iou_per_person = [0 for i in l_bbox_person]
        status_found = False
        image_name = imageid2imagename[image_id]
        if len(l_bbox_motorbike) > 0:
            l_image_have_motorbike.append(image_name)
    print("Number images having motorbike: ",len(l_image_have_motorbike))
    # file_log_motorbike = open("/home/asilla/duongnh/project/Analys_COCO/Output/log_cat.txt",'w')
    # for file in l_image_have_motorbike:
    #     file_log_motorbike.write(file+'\n')
    # file_log_motorbike.close()
    l_overlap = []
    for file_motor_source in l_image_have_motorbike:
        if file_motor_source in l_motorbike_npy:
             l_overlap.append(file_motor_source)
    print("Number images overlap: ",len(l_overlap))
    np.save("overlap.npy", l_overlap)