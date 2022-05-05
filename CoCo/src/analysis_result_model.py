import json 
import numpy as np 
import os 
import time 


def load_json_result(path_json):
    file_annotation= open(path_json,'r')
    l_annotation = json.load(file_annotation)
    dict_image_id = {}
    for annotation in l_annotation:
        image_id = annotation["image_id"]
        keypoint = annotation["keypoints"]
        # print(image_id)
        if image_id not in dict_image_id:
            dict_image_id[image_id] = [keypoint]
        else:
            dict_image_id[image_id].append(keypoint)
    return dict_image_id


def load_groundtruth(path_json):
    '''
    '''
    file_annotation_person = open(path_json,'r')
    json_data_person = json.load(file_annotation_person)
    l_info_person = json_data_person['info']
    l_image_person = json_data_person['images']
    l_annotations_person = json_data_person['annotations']
    l_categories_person = json_data_person['categories']
    l_image_id_annotation_person = {}
    for annotation_person in l_annotations_person:
        keypoints = annotation_person['keypoints']
        image_id = annotation_person['image_id']
        if image_id not in l_image_id_annotation_person:
            l_image_id_annotation_person[image_id] = [keypoints]
        else:
            l_image_id_annotation_person[image_id].append(keypoints)

    dict_imageid_person = {}
    for annotation_person in l_image_person:
        file_name_person = annotation_person['file_name']
        # print(file_name_person)
        image_id_person = annotation_person['id']
        dict_imageid_person[file_name_person] = image_id_person
    return l_image_id_annotation_person, dict_imageid_person

if __name__ == '__main__':
    path_data = "/home/asilla/duongnh/datasets/annotations/person_keypoints_train2017_old.json"
    path_result = "/home/asilla/duongnh/project/alphapose-results.json"
    dict_image_id_annotation_gt, dict_imageid_person = load_groundtruth(path_data)
    dict_image_id_rs = load_json_result(path_result)

    for file_name in dict_image_id_rs.keys():
        # if file_name in dict_imageid_person:
        image_id = dict_imageid_person[file_name]
        if file_name in dict_image_id_rs:
            print(file_name)
            l_keypoint_rs = dict_image_id_rs[file_name]
            l_keypoint_gt = dict_image_id_annotation_gt[image_id]
            print(len(l_keypoint_rs), len(l_keypoint_gt))
        else:
            print(file_name)
    