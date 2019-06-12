"""The Caltech-UCSD bird dataset
"""

import numpy as np
import os
from scipy import misc
from skimage import io
import ipdb
import shutil
import random

np.random.seed(0)
random.seed(1)

class CUBDataLayer():
    """ The Caltech-UCSD bird dataset
    """
    def __init__(self, **kwargs):
        """Load the dataset.
        kwargs:
            root: the root folder of the CUB_200_2011 dataset.
            is_training: if true, load the training data. Otherwise, load the
                testing data.
            crop: if None, does not crop the bounding box. If a real value,
                crop is the ratio of the bounding box that gets cropped.
                e.g., if crop = 1.5, the resulting image will be 1.5 * the
                bounding box area.
            target_size: all images are resized to the size specified. Should
                be a tuple of two integers, like [256, 256].
            version: either '2011' or '2010'.
        Note that we will use the python indexing (labels start from 0).
        """
        root = '/home/huaxiuyao/Data/meta-dataset/CUB_Bird/CUB_200_2011/'

        crop = True
        target_size = [84,84]
        images = [line.split()[1] for line in
                    open(os.path.join(root, 'images.txt'), 'r')]
        boxes = [line.split()[1:] for line in
                    open(os.path.join(root, 'bounding_boxes.txt'),'r')]


        # for the boxes, we store them as a numpy array
        boxes = np.array(boxes, dtype=np.float32)
        boxes -= 1
        # load the data
        self._load_data(root, images, boxes, crop, target_size)

    def _load_data(self, root, images, boxes, crop, target_size):
        num_imgs = len(images)

        for i in range(num_imgs):
            image = io.imread(os.path.join(root, 'images', images[i]))
            if image.ndim == 2:
                image = np.tile(image[:,:,np.newaxis], (1, 1, 3))
            if image.shape[2] == 4:
                image = image[:, :, :3]
            if crop:
                image = self._crop_image(image, crop, boxes[i])
            data_img = misc.imresize(image, target_size)
            misc.imsave(os.path.join(root, 'images', images[i]), data_img)

            if i%500==0:
                print(i)

        return

    def _crop_image(self, image, crop, box):
        imheight, imwidth = image.shape[:2]
        x, y, width, height = box
        centerx = x + width / 2.
        centery = y + height / 2.
        xoffset = width * crop / 2.
        yoffset = height * crop / 2.
        xmin = max(int(centerx - xoffset + 0.5), 0)
        ymin = max(int(centery - yoffset + 0.5), 0)
        xmax = min(int(centerx + xoffset + 0.5), imwidth - 1)
        ymax = min(int(centery + yoffset + 0.5), imheight - 1)
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            raise ValueError("The cropped bounding box has size 0.")
        return image[ymin:ymax, xmin:xmax]

def select_image():
    path='/home/huaxiuyao/Data/meta-dataset/CUB_Bird/images/'
    dirlist=os.listdir(path)
    num_images=[]
    for eachdir in dirlist:
        tmp=os.listdir(path+eachdir)
        for each in tmp:
            if each[0]=='.':
                print(eachdir, each)
        if len(os.listdir(path+eachdir))==60:
            num_images.append([eachdir, len(os.listdir(path+eachdir))])
    all_folder_id=random.sample(range(len(num_images)), 100)
    all_folder=[num_images[id] for id in all_folder_id]
    random.shuffle(all_folder)
    for i in range(64):
        shutil.move(path+all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/CUB_Bird/train/')
    for i in range(64,80):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/CUB_Bird/val/')
    for i in range(80,100):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/CUB_Bird/test/')
    # num_images=sorted(num_images, key=lambda x:x[1], reverse=True)

if __name__=='__main__':
    select_image()