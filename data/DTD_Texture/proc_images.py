"""
Usage instructions:
    First download the omniglot dataset
    and put the contents of both images_background and images_evaluation in data/omniglot/ (without the root folder)

    Then, run the following:
    cd data/
    cp -r omniglot/* omniglot_resized/
    cd omniglot_resized/
    python resize_images.py
"""
from PIL import Image
import glob
import os
import numpy as np
import random
import shutil

np.random.seed(0)
random.seed(1)

def Process():
    image_path = '/home/huaxiuyao/Data/meta-dataset/DTD_Texture/dtd/images/*/'

    all_images = glob.glob(image_path + '*')

    i = 0

    for image_file in all_images:
        im = Image.open(image_file)
        im = im.resize((84,84), resample=Image.LANCZOS)
        im.save(image_file)
        i += 1

        if i % 200 == 0:
            print(i)

def select_image():
    path = '/home/huaxiuyao/Data/meta-dataset/DTD_Texture/images/'
    dirlist = os.listdir(path)
    num_images = []
    for eachdir in dirlist:
        num_images.append([eachdir, len(os.listdir(path + eachdir))])
    all_folder_id = random.sample(range(len(num_images)), 47)
    all_folder = [num_images[id] for id in all_folder_id]
    random.shuffle(all_folder)
    for i in range(30):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/DTD_Texture/train/')
    for i in range(30, 37):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/DTD_Texture/val/')
    for i in range(37, 47):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/DTD_Texture/test/')
    # num_images = sorted(num_images, key=lambda x: x[1], reverse=True)

if __name__=='__main__':
    select_image()