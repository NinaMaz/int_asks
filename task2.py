import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from torchvision import transforms
from PIL import Image


def to_numpy(tensor_img):
    return tensor_img[0].permute(1, 2, 0).numpy()

def to_tensor(tensor_img):
    patch = torch.tensor(tensor_img)
    return patch.permute(2, 0, 1).unsqueeze(0)

def read_image(path):
    img = transforms.PILToTensor()(Image.open(path)).unsqueeze(0)/255 
    return img

def write_image(img, path):
    Image.fromarray((img* 255 / np.max(img)).astype(np.uint8), 'RGB').save(path)
    return
    
def remove_horizontal_lines(img):
    # convolution with vertical edge detector kernel
    kernel = torch.ones(3, 3, 3, 3) * torch.tensor([-1, 0, 1])
    convolved = F.conv2d(img, kernel, padding='same').clamp(0, 1)
    
    #make binary mask
    threshold = 0.6
    clamped = (convolved > threshold).to(float)
    
    # use dilation from cv2 to dilate the brightest pixels
    ks = 5
    kernel = np.ones((ks,ks),np.uint8)
    dilation = cv2.dilate(clamped[0, 0, :, :].numpy(),kernel,iterations = 1)
    
    # erosion+dilation to create a prominent vertical line mask
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
    detected_lines = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    mask = torch.tensor(np.asarray([detected_lines]*3)).unsqueeze(0)
    clean = (img + mask).clamp(0, 1).float()

    # dilation+erosion to remove noise
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,1))
    result = 255 - cv2.morphologyEx(255 - to_numpy(clean), cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    
    return result



parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Input file path")
parser.add_argument("-o", "--out", help="Output file path", default="result.jpg")

opts = parser.parse_args()

module_path = os.path.dirname(os.path.realpath(__file__))

input_path = os.path.join(module_path, opts.file)
output_path = os.path.join(module_path, opts.out)
print("Input: ", input_path)
print("Output: ", output_path)

img = read_image(input_path)
res = remove_horizontal_lines(img)
write_image(res, output_path)
    
    