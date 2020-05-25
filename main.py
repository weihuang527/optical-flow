import os
import cv2
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from gen_flow import gen_flow_circle
from flow_display import dense_flow, sparse_flow
from image_warp import image_warp

if __name__ == "__main__":
    case = 2  # 1 or 2
    height = 101
    width = 101
    if case == 1:
        flow = gen_flow_circle([height//2, width//2], height, width)
        flow = flow / 3
    else:
        flow = gen_flow_circle([0,0], height, width)
        flow = flow / 5

    print(flow[:,:,0])
    print(flow[:,:,1])

    dense_flow(flow)
    sparse_flow(flow, stride=10)

    img = np.zeros((height, width), dtype=np.uint8)
    img = cv2.circle(img, (width//2, height//2), radius=25, color=255, thickness=1)

    deformed_nearest = image_warp(img.copy(), flow, mode='nearest')  # nearest or bilinear
    deformed_bilinear = image_warp(img.copy(), flow, mode='bilinear')

    img_cat = np.concatenate([img, deformed_nearest, deformed_bilinear], axis=1)
    _, w = img_cat.shape
    img_cat[:, width] = 255
    img_cat[:, width*2] = 255
    Image.fromarray(img_cat).save('./deformed.png')