import cv2
import glob
import torch
import argparse
import numpy as np
import matplotlib.image as pltimg
from os import path
from models import *

IMG_HEIGHT, IMG_WIDTH = 224, 224
BGR_MEAN = np.array([[[102.9801, 115.9465, 122.7717]]])
NUM_OF_POINTS = {"upper": 6, "lower": 4, "full": 8}
VIS_CASE = ['Visible','Occlude','Inexistent']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch setup: {} is available".format(device))

def load_image_resize_asp(img_path, dsize = (224,224)): # in 255 scale, shape = dsize[0] * dsize[1] * 3 
    img = cv2.imread(img_path)
    if np.mean(img) < 1:
        img = (img*255)
    img = img.astype(np.uint8)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis = 2)
    elif len(img.shape) > 3:
        print("Bad shape.")

    if img.shape[-1] == 1 :
        img = np.tile(img, (1,1,3))
    elif img.shape[-1] == 4:
        img = img[:,:,:3]
    elif img.shape[-1] == 3: 	
        pass
    else:
        print("Bad channel {}".format(img_path))
    
    scale = dsize[0]/ max(img.shape);
    s1 = round(img.shape[0]*scale);
    s2 = round(img.shape[1]*scale);
    img_resized = None
    offset = [0,0,0]
    if not(img.shape[0] == 224 and img.shape[1] == 224):
        img_resized = cv2.resize(img,(s1,s2));
        pad = np.array([224,224, 0]) - img_resized.shape
        pad[pad < 0] = 0
        offset = np.floor(pad/2).astype(np.uint8)     
        img = cv2.copyMakeBorder(img_resized, offset[0], (pad-offset)[0], offset[1], (pad-offset)[1], cv2.BORDER_CONSTANT, value = [0,0,0])
    resize_info = [scale, offset]
    return img, resize_info 



def load_models(path_to_model_dir, option):
    models = []
    
    if option == "upper":
        print("Loading {} model".format(option ))
        fld = FLD_u1()
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage1.caffemodel.pt".format(option)))) 
        fld.to(device)
        fld.eval()
        models.append(fld)
        print("Stage 1 loading successful.")
        fld = FLD_u2()
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage2.caffemodel.pt".format(option))))
        fld.to(device) 
        fld.eval()
        models.append(fld)
        print("Stage 2 loading successful.")
        fld = FLD_u3(20)
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage3_easy.caffemodel.pt".format(option)))) 
        fld.to(device)  
        fld.eval()   
        models.append(fld)
        print("Stage 3e loading successful.")
        fld = FLD_u3(20)
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage3_hard.caffemodel.pt".format(option)))) 
        fld.to(device)   
        fld.eval()  
        models.append(fld)
        print("Stage 3h loading successful.")
    
    if option == "lower":
        print("Loading {} model".format(option ))
        fld = FLD_l1()
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage1.caffemodel.pt".format(option)))) 
        fld.to(device) 
        fld.eval()    
        models.append(fld)
        print("Stage 1 loading successful.")
        fld = FLD_l2()
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage2.caffemodel.pt".format(option)))) 
        fld.to(device) 
        fld.eval()
        models.append(fld)
        print("Stage 2 loading successful.")
        fld= FLD_l3(64)
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage3_easy.caffemodel.pt".format(option)))) 
        fld.to(device) 
        fld.eval()
        models.append(fld)
        print("Stage 3e loading successful.")
        fld = FLD_l3(64)
        fld.load_state_dict(torch. load(path.join(path_to_model_dir, "FLD_{}/stage3_hard.caffemodel.pt".format(option)))) 
        fld.to(device)
        fld.eval() 
        models.append(fld)
        print("Stage 3h loading successful.")

    if option == "full":
        print("Loading {} model".format(option))
        fld = FLD_f1()
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage1.caffemodel.pt".format(option)))) 
        fld.to(device) 
        fld.eval()
        models.append(fld)
        print("Stage 1 loading successful.")
        fld = FLD_f2()
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage2.caffemodel.pt".format(option)))) 
        fld.to(device) 
        fld.eval()
        models.append(fld)
        print("Stage 2 loading successful.")
        fld = FLD_f3(256)
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}stage3_easy.caffemodel.pt".format(option)))) 
        fld.to(device)
        fld.eval() 
        models.append(fld)
        print("Stage 3e loading successful.")
        fld = FLD_f3(128)
        fld.load_state_dict(torch.load(path.join(path_to_model_dir, "FLD_{}/stage3_hard.caffemodel.pt".format(option)))) 
        fld.to(device) 
        fld.eval()
        models.append(fld)
        print("Stage 3h loading successful.")

    return models


def preprocess(img):
    img = img.astype(np.float32)
    img -= BGR_MEAN
    img = np.transpose(img, (2,0,1))  # x/ y transpose??
    #expand dim
    img = np.expand_dims(img, axis = 0)
    #to pytorch tensor
    img = torch.from_numpy(img)
    img = img.to(device)
    return img

	

def detect_landmark(img, models, resize_info):
    
    scale, offset = resize_info
    get_orig_coordinate = lambda p: ((p+0.5)*224 - np.tile(np.transpose([[offset[1],offset[0]]]),[NUM_OF_POINTS[option],1]))/scale;

    # stage 1 fp
    stage_1_net = models[0].eval()
    res_stage1 = stage_1_net(img)
    landmark_stage1 = res_stage1[0][0:NUM_OF_POINTS[option]*2]  
    v1 = np.argmax(np.reshape(res_stage1[0][NUM_OF_POINTS[option]*2:].cpu().detach().numpy(), (3,NUM_OF_POINTS[option])), axis = 0)
    visibility_stage1 = [ VIS_CASE[x] for x in v1]
    landmark_coor1 = np.reshape(landmark_stage1.cpu().detach().numpy(), (NUM_OF_POINTS[option]*2,1))
    prediction_stage1 = [get_orig_coordinate(landmark_coor1), visibility_stage1]
    # stage 2 fp   
    stage_2_net = models[1].eval()
    res_stage2 = stage_2_net(img, landmark_stage1)
    landmark_stage2 = landmark_stage1-res_stage2[0][0:NUM_OF_POINTS[option]*2]/5
    v2 = np.argmax(np.reshape(res_stage2[0][NUM_OF_POINTS[option]*2:].cpu().detach().numpy(), (3,NUM_OF_POINTS[option])), axis = 0)
    visibility_stage2 = [ VIS_CASE[x] for x in v2]
    landmark_coor2 = np.reshape(landmark_stage2.cpu().detach().numpy(), (NUM_OF_POINTS[option]*2,1))
    prediction_stage2 = [get_orig_coordinate(landmark_coor2), visibility_stage2]
    # stage 3 fp
    stage_3_easy = models[2].eval()
    stage_3_hard = models[3].eval()
    res_stage3_easy = stage_3_easy(img,landmark_stage2)
    res_stage3_hard = stage_3_hard(img,landmark_stage2)
    landmark_stage3 = landmark_stage2-(res_stage3_easy[0][0:NUM_OF_POINTS[option]*2]/5 + res_stage3_hard[0][0:NUM_OF_POINTS[option]*2]/5)/2;
    
    v3 = np.argmax(np.reshape(res_stage3_easy[0][NUM_OF_POINTS[option]*2:].cpu().detach().numpy() + res_stage3_hard[0][NUM_OF_POINTS[option]*2:].cpu().detach().numpy(), \
                             (3,NUM_OF_POINTS[option])), axis = 0)

    visibility_stage3 = [VIS_CASE[x] for x in v3]
    landmark_coor3 = np.reshape(landmark_stage3.cpu().detach().numpy(), (NUM_OF_POINTS[option]*2,1))
    prediction_stage3 = [get_orig_coordinate(landmark_coor3), visibility_stage3]
    

    return [prediction_stage1,prediction_stage2,prediction_stage3]
    

def visualize(img, predictions, save_path = None):
    for i in range(len(predictions)):
        p = predictions[i]
        if i == 2:
            color = (0,0,255) 
        #elif i == 1:
        #    color = (0,255,0) 
        else:
            continue      
        for vc in range(3):
            for i in range(len(p[1])):
                if p[1][i] == VIS_CASE[vc]:
                    x,y = p[0][2*i][0], p[0][2*i + 1][0]
                    img = cv2.circle(img, (int(x), int(y)), radius= 2*(3-vc), color=color, thickness=-1)
    if save_path:
        print(save_path)
        cv2.imwrite(save_path, img)
    

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("image_path")
parser.add_argument("output_path")
args = parser.parse_args()

d_img = path.join(args.image_path)+"/*"
img_paths = sorted(glob.glob(d_img))
option = "upper"

models = load_models(args.model_path, option)

i = 0
for img_path in img_paths:
    img, resize_info = load_image_resize_asp(img_path, dsize = (IMG_HEIGHT, IMG_WIDTH))
    img_tensor = preprocess(img)
    predictions = detect_landmark(img_tensor, models, resize_info)
    visualize(img, predictions, save_path = args.output_path + "/"+ img_path.split("/")[-2] + "/landmark_" + img_path.split("/")[-1] )
    i+=1



