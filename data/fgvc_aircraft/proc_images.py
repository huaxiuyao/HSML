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
import scipy.io as scio
import os
from scipy import misc
from skimage import io
import ipdb
import shutil
import random

np.random.seed(1)
random.seed(2)

class FGVC_Aircraft():
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
        root = '/home/huaxiuyao/Data/meta-dataset/FGVC_Aircraft/data/'

        crop = True
        target_size = [84,84]
        images = [imageid.split('.')[0] for imageid in os.listdir(root+'images')]
        boxes = {line.split()[0]:line.split()[1:] for line in
                    open(os.path.join(root, 'images_box.txt'),'r')}


        # for the boxes, we store them as a numpy array
        for eachkey in boxes:
            boxes[eachkey] = np.array(boxes[eachkey], dtype=np.float32) - 1
        # load the data
        self._load_data(root, images, boxes, crop, target_size)

    def _load_data(self, root, images, boxes, crop, target_size):
        num_imgs = len(images)

        for i in range(num_imgs):
            image = io.imread(os.path.join(root, 'images', '{}.jpg'.format(images[i])))
            if image.ndim == 2:
                image = np.tile(image[:,:,np.newaxis], (1, 1, 3))
            if image.shape[2] == 4:
                image = image[:, :, :3]
            if crop:
                image = self._crop_image(image, crop, boxes[images[i]])
            data_img = misc.imresize(image, target_size)
            misc.imsave(os.path.join(root, 'images', '{}.jpg'.format(images[i])), data_img)

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

def reorganize():
    root='/home/huaxiuyao/Data/meta-dataset/FGVC_Aircraft/data/'
    label=[line.strip().split(' ') for line in open(os.path.join(root, 'images_variant_train.txt'),'r')]
    label.extend([line.strip().split(' ') for line in open(os.path.join(root, 'images_variant_trainval.txt'),'r')])
    label.extend([line.strip().split(' ') for line in open(os.path.join(root, 'images_variant_val.txt'), 'r')])
    label.extend([line.strip().split(' ') for line in open(os.path.join(root, 'images_variant_test.txt'), 'r')])
    labelall={}
    for eachitem in label:
        if eachitem[0] in labelall:
            continue
        labelall[eachitem[0]]='-'.join(eachitem[1:])
    newpath = '/home/huaxiuyao/Data/meta-dataset/FGVC_Aircraft/data/organized_images/'
    for eachfile in os.listdir('/home/huaxiuyao/Data/meta-dataset/FGVC_Aircraft/data/images/'):
        tmp_id = eachfile.split('.')[0]
        folder_id = labelall[tmp_id]
        print(folder_id)
        if folder_id == 'F-16A/B':
            folder_id='F-16A-B'
        if folder_id == 'F/A-18':
            folder_id='F-A-18'
        if not os.path.isdir(newpath + '{}'.format(folder_id)):
            os.mkdir(newpath + '{}'.format(folder_id))

        image_file = '/home/huaxiuyao/Data/meta-dataset/FGVC_Aircraft/data/images/' + eachfile
        im = Image.open(image_file)
        im.save(newpath + '{}'.format(folder_id) + '/' + eachfile)

def select_image():
    path='/home/huaxiuyao/Data/meta-dataset/FGVC_Aircraft/images/'
    dirlist=os.listdir(path)
    num_images=[]
    for eachdir in dirlist:
        num_images.append([eachdir, len(os.listdir(path+eachdir))])
    all_folder_id=random.sample(range(len(num_images)), 100)
    all_folder=[num_images[id] for id in all_folder_id]
    random.shuffle(all_folder)
    for i in range(64):
        shutil.move(path+all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/FGVC_Aircraft/train/')
    for i in range(64,80):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/FGVC_Aircraft/val/')
    for i in range(80,100):
        shutil.move(path + all_folder[i][0], '/home/huaxiuyao/Data/meta-dataset/FGVC_Aircraft/test/')
    # num_images=sorted(num_images, key=lambda x:x[1], reverse=True)


if __name__=='__main__':
    select_image()