import numpy as np 
import cv2 
import os 



path_folder_retina = "visualize"
path_folder_hrnet = "visualize_hrnet"
path_folder_concat_image = "prepare_retina_hrnet"

font = cv2.FONT_HERSHEY_SIMPLEX
  ss
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2
   
# Using cv2.putText() method

for n_image in os.listdir(path_folder_retina):
    path_image_retina = path_folder_retina + "/" + n_image 
    path_image_hrnet = path_folder_hrnet + "/" + n_image 
    image_draw_retina = cv2.imread(path_image_retina)
    image_draw_hrnet = cv2.imread(path_image_hrnet)
    image_draw_retina = cv2.putText(image_draw_retina, 'Retina', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    image_draw_hrnet = cv2.putText(image_draw_hrnet, 'HRNet', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    im_h = cv2.hconcat([image_draw_retina, image_draw_hrnet])
    cv2.imwrite(path_folder_concat_image + "/" + n_image, im_h)
    