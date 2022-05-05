import os 
import numpy as np 
import time 
import random

path_dataset = "/home/asilla/duongnh/datasets/training_usage/coco2017train_after_filter.txt"

dataset = open(path_dataset, 'r')
l_item = []
for line in dataset:
    line = line[:-1]
    l_item.append(line)

random.shuffle(l_item)

num_sample = 500