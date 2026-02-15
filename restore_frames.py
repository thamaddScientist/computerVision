import cv2
import os
import numpy as np


EXTRACTED_FRAMES_DIRECTORY = './extracted-frames'

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

SHARPEN_KERNEL = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])


def restore_frame(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image at path {image_path}")
        return None
    
    #denoising images
    denoised = cv2.GaussianBlur(img,(3,3),0)

    #convert to lab color space to isolate L channel
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    #applying CLAHE to l channel
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,tileGridSize=CLAHE_GRID_SIZE)
    cl = clahe.apply(l_channel)

    #merging it all back together
    limg = cv2.merge((cl,a_channel,b_channel))
    orig_color = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    #applying sharpening filter to image
    sharpened_image = cv2.filter2D(orig_color,-1,SHARPEN_KERNEL)

    return sharpened_image


def restoration_engine(frames_path,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if not os.path.exists(frames_path):
        print(f"Frames path does not exist: {frames_path}")
        return 
    
    image_files = os.listdir(frames_path)

    for image in image_files:
        image_path = os.path.join(frames_path,image)
        restored_image = restore_frame(image_path)

        cv2.imwrite(f"{output_path}/{image}",restored_image)

    print(f"Restoration Complete, {len(image_files)} images restored.")

    
    
if __name__=='__main__':
    restoration_engine(EXTRACTED_FRAMES_DIRECTORY,'./restored-images')

