from json.tool import main
import cv2
from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
from SimpleHRNET import SimpleHRNet
import torch 
import os 

import numpy as np 

def visualize_keypoint(image, keypoints, color):
    for i in range(17):
        single_kp = keypoints[i]
        x , y , confidence_score = single_kp
        x = int(x)
        y = int(y)
        if x != 0 and y != 0:
            image = cv2.circle(image, (x,y), 5,color, -1)
    return image

def default_visualize(image, joints, color):
    for j in range(len(joints)):
        points = []
        Y = []
        X = []
        for i, coords in enumerate(joints[j]):
            y, x, _ = coords
            points.append([x,y])
            Y.append(y)
            X.append(x)
            #print(Y)
            image = cv2.circle(image, (int(x),int(y)), radius=3, color=color, thickness=-1)
    return image



class HumanPose_Estimation:
    def __init__(self):
        self.path_pretrained = "/home/asilla/duongnh/project/simple-HRNet/pose_hrnet_w48_384x288.pth"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = (384, 288) # (height, width)
        self.model = SimpleHRNet(48, 17, multiperson = False,\
                                checkpoint_path = self.path_pretrained, device = self.device)
    def predict(self, image):
        status_detection = False
        joints = self.model.predict(image)
        if len(joints) == 0:
            return (0,0), status_detection
        status_detection = True
        human_joints = joints[0]
        y, x, confidence_score  = human_joints[0]
        nose_coordinate = (int(x), int(y))
        return nose_coordinate, status_detection, confidence_score
        
        
        
# if __name__ == '__main__':
#     device = torch.device("cuda")
#     model = SimpleHRNet(48, 17,multiperson= False,  checkpoint_path = "pose_hrnet_w48_384x288.pth", device=device)
#     # path_folder_input = "/home/asilla/duongnh/project/Analys_COCO/crop_image"
#     # # count = 0
#     # for file in os.listdir(path_folder_input)[:100]:

#     path_image = "/home/asilla/duongnh/datasets/CrownPose/images/119666.jpg" 
    
#     image = cv2.imread(path_image)
#     # image = cv2.resize(image, (288, 384))
#     print(np.shape(image))
#     joints = model.predict(image)
#     # print(joints)
#     # kp = joints[0]
#     color = (0,0,255)
#     image = default_visualize(image, joints, color)
#     cv2.imwrite("result_inference/" + path_image.split("/")[-1], image)
#         # count += 1
