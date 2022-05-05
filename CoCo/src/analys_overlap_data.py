import os 
import time 
import numpy as np 
from tqdm import tqdm


#Step 1: Cuong vs after

# dataset_after = open("/home/asilla/duongnh/datasets/training_usage/coco2017train_after_filter.txt",'r')

before_after_dict_image = "/home/asilla/duongnh/datasets/Filtered_data_29_03/Log_before_after"
duong_dic_image = "/home/asilla/duongnh/datasets/Filtered_data_29_03/Log_bbox_48_pixel_allcoco_filtered"
des_folder = "/home/asilla/duongnh/datasets/Filtered_data_29_03/Log_beforeafter_overlap_duong_filtered"
path_source_image = "/home/asilla/duongnh/datasets/train2017"


if not os.path.exists(des_folder):
    os.mkdir(des_folder)
l_after = os.listdir(before_after_dict_image)
l_duong = os.listdir(duong_dic_image)
# for line in dataset_after:
#     line_after = line[:-1]
#     l_after.append(line_after)




count_overlap = 0
for item in tqdm(l_after):
    if item in l_duong:
        count_overlap += 1

        path_image = path_source_image +"/"+ item
        des_image = des_folder + "/" + item
        command_line = "cp {} {}".format(path_image, des_image)
        os.system(command_line)

print(count_overlap)
# for item in tqdm(os.listdir(des_folder)):
#     # if item in l_after:
#     path_image = des_folder +"/"+ item
#     des_image = source_image + "/" + item
#     command_line = "cp {} {}".format(path_image, des_image)
#     os.system(command_line)