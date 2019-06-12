from PIL import Image
import glob
import os
import shutil
import random
import numpy as np
import ipdb

np.random.seed(1)
random.seed(2)

image_path = '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/images/*/'


def process():
    all_images = glob.glob(image_path + '*')

    i = 0

    for image_file in all_images:
        im = Image.open(image_file)
        im = im.resize((84, 84), resample=Image.LANCZOS)
        im.save(image_file)
        i += 1

        if i % 200 == 0:
            print(i)


def select_folder():
    path = '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/images/'
    dirlist = os.listdir(path)
    num_images = []
    for eachdir in dirlist:
        if len(os.listdir(path + eachdir)) >= 150:
            num_images.append([eachdir, len(os.listdir(path + eachdir))])
    all_folder_id = random.sample(range(len(num_images)), 100)
    all_folder = [num_images[id] for id in all_folder_id]
    random.shuffle(all_folder)
    for i in range(64):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/train/')
    for i in range(64, 80):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/val/')
    for i in range(80, 100):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/test/')
    # num_images = sorted(num_images, key=lambda x: x[1], reverse=True)
    # print(len(num_images))


def select_image():
    folder = ['train', 'test', 'val']
    for eachfolder in folder:
        all_files = os.listdir('/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/{}/'.format(eachfolder))
        for eachtype in all_files:
            images = os.listdir('/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/{}/{}/'.format(eachfolder, eachtype))
            random.shuffle(images)
            images_id = random.sample(range(len(images)), 150)
            new_images = [images[idx] for idx in images_id]
            os.mkdir('/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/{}_new/{}/'.format(eachfolder, eachtype))
            for idx_y in range(len(new_images)):
                shutil.move('/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/{}/{}/{}'.format(eachfolder, eachtype,
                                                                                         new_images[idx_y]),
                            '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/{}_new/{}/'.format(eachfolder, eachtype))
    # path = '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/images/'
    # dirlist = os.listdir(path)
    # num_images = []
    # for eachdir in dirlist:
    #     if len(os.listdir(path + eachdir)) >= 150:
    #         num_images.append([eachdir, len(os.listdir(path + eachdir))])
    # all_folder_id = random.sample(range(len(num_images)), 100)
    # all_folder = [num_images[id] for id in all_folder_id]
    # random.shuffle(all_folder)
    # for i in range(64):
    #     shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/train/')
    # for i in range(64, 80):
    #     shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/val/')
    # for i in range(80, 100):
    #     shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/FGVCx_Fungi/test/')
    # num_images = sorted(num_images, key=lambda x: x[1], reverse=True)
    # print(len(num_images))


if __name__ == '__main__':
    select_folder()
