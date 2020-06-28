"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
from time import gmtime, strftime
from matplotlib.pyplot import cm
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csvTools
import cv2
import glob
from random import shuffle

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_A = img_A[...,np.newaxis]
    img_B = img_B[...,np.newaxis]
    img_AB = np.concatenate((img_A, img_B), axis=-1)
    print(img_AB.shape)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB #  256, 256, 2

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=257, fine_size=257, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


class LIDC(object):

    def __init__(self):

        self.dataname = "lidc"
        self.dims = 112*112
        self.shape = [112 , 112 , 1]
        self.image_size = 112
        self.load_lidc()
        #self.data, self.data_y = self.load_lidc()

    def load_lidc(self):
        '''
        mnist data size:
            (70000, 28, 28, 1)
            (70000,)
        '''
        data_dir = os.path.join('./huimages/')
        mask_dir = os.path.join('./masks/')
        noduleinfo = csvTools.readCSV('files/malignancy.csv')

        self.images_list = os.listdir(data_dir)
        self.masks_list = os.listdir(mask_dir)
        self.noduleinfo = noduleinfo

        print('number of images_list: ', len(self.images_list))
        print('number of mask_dir: ', len(self.masks_list))
        print('number of noduleinfo: ', len(self.noduleinfo))

        trainingdata = []
       
        for onenodule in noduleinfo:
            scanid = onenodule[1]
            scanid = caseid_to_scanid(int(scanid))
            noduleid = onenodule[3]
            scan_list_id = onenodule[2]

            nodule_image_name = str(scanid) + '_' + str(noduleid) + '_' + str(scan_list_id) + '.npy'
            if nodule_image_name in self.images_list:
                # lobulation = onenodule[26]
                lobulation = onenodule[28]
                spiculation = onenodule[27]
                malignancy = onenodule[29]

                if float(lobulation) >= 0 or float(spiculation) >= 0:
                    trainingdata.append(onenodule)
                    # print(len(onenodule)) # 30 
                    ## TODO 数据扩充
                if float(lobulation) >= 3 or float(spiculation) >= 3:
                    import copy
                    copy.deepcopy
                    left = copy.deepcopy(onenodule)
                    right = copy.deepcopy(onenodule)
                    down = copy.deepcopy(onenodule)
                    left.append('left')
                    right.append('right')
                    down.append('down')
                    trainingdata.append(left)
                    trainingdata.append(right)
                    trainingdata.append(down)
                    # print(len(left))  # 31      

        
        print('number of training data: ', len(trainingdata)) 
        trainingdata = np.array(trainingdata)
        shuffle(trainingdata)
        self.data = trainingdata
        '''
         LIDC-IDRI-1011_4_2.npy
        '''

    def getNext_batch(self, iter_num=0, batch_size=64):

        ro_num = len(self.data) / batch_size - 1

        # if iter_num % ro_num == 0:
        batch_data = self.data[int(iter_num % ro_num) * batch_size: int(iter_num% ro_num + 1) * batch_size]
        # batch_data = self.data
        length = len(batch_data)
        # print(int(iter_num % ro_num) * batch_size)
        # print(int(iter_num% ro_num + 1) * batch_size)

        labels = []
        images = []
        masks = []
        lungs = []
        mediastinums = []
        for onedata in batch_data:
            # float(onedata[26]

            scanid = onedata[1]
            scanid = caseid_to_scanid(int(scanid))
            noduleid = onedata[3]
            scan_list_id = onedata[2]

            nodule_image_name = str(scanid) + '_' + str(noduleid) + '_' + str(scan_list_id)
            # print(nodule_image_name)

            nodule_npy = nodule_image_name + '.npy'
            # if len(onedata)==30:
            hu = np.load('./huimages/' + nodule_npy)
            image = normalization(hu)
            image = cv2.resize(image, (128, 128))

            hu = np.load('./huimages/' + nodule_npy)
            lung = truncate_hu(hu, 50, -1250)
            # lung = normalization(lung)
            lung = cv2.resize(lung, (128, 128))

            hu = np.load('./huimages/' + nodule_npy)
            mediastinum = truncate_hu(hu,240,-160)
            # mediastinum = normalization(mediastinum)
            mediastinum = cv2.resize(mediastinum, (128, 128))
            # make bone  imfomation
            bone = np.load('./huimages/' + nodule_npy)
            bone[bone<=300] = 0 
            bone[bone>300] = 1
             
            mask = np.load('./masks/' + nodule_npy) + bone
            mask[mask>=1]=1
            # bone = cv2.resize(bone, (128, 128))
            # plt.imsave('bone_mask/'+str(nodule_image_name)+'.jpg',bone,cmap=cm.gray) 
            mask = cv2.resize(mask, (128, 128))
            x, y = 63.5,63.5
            if onedata[-1] =='left':
                matRoate = cv2.getRotationMatrix2D((x, y), 90, 1.)
                image = cv2.warpAffine(image, matRoate, (128, 128))
                lung = cv2.warpAffine(lung, matRoate, (128, 128))
                mediastinum = cv2.warpAffine(mediastinum, matRoate, (128, 128))
                mask = cv2.warpAffine(mask, matRoate, (128, 128))
                # print('left')

            if onedata[-1] =='right':
                matRoate = cv2.getRotationMatrix2D((x, y), -90, 1.)
                image = cv2.warpAffine(image, matRoate, (128, 128))
                lung = cv2.warpAffine(lung, matRoate, (128, 128))
                mediastinum = cv2.warpAffine(mediastinum, matRoate, (128, 128))
                mask = cv2.warpAffine(mask, matRoate, (128, 128))
                # print('right')

            if onedata[-1] =='down':
                matRoate = cv2.getRotationMatrix2D((x, y), 180, 1.)
                image = cv2.warpAffine(image, matRoate, (128, 128))
                lung = cv2.warpAffine(lung, matRoate, (128, 128))
                mediastinum = cv2.warpAffine(mediastinum, matRoate, (128, 128))
                mask = cv2.warpAffine(mask, matRoate, (128, 128))
                # print('down')

                

            images.append(image)
            lungs.append(lung)
            mediastinums.append(mediastinum)
            masks.append(mask)

            # lobulation = np.array(one_shot_attri(float(onedata[26])))
            lobulation = np.array(one_shot_attri(float(onedata[28])))
            spiculation = np.array(one_shot_attri(float(onedata[27])))
            malignancy = np.array(one_shot_attri(float(onedata[29])))
            y = np.concatenate((lobulation, spiculation, malignancy), axis = 0)
            # print(y.shape)

            labels.append(y)
            # print(y)
        images = np.expand_dims(images, axis=3)
        masks = np.expand_dims(masks, axis=3)
        lungs = np.expand_dims(lungs, axis = 3)
        mediastinums = np.expand_dims(mediastinums, axis = 3)

        return images, lungs, mediastinums, masks, labels
        #     perm = np.arange(length)
        #     np.random.shuffle(perm)
        #     self.data = np.array(self.data)
        #     self.data = self.data[perm]
        #     self.data_y = np.array(self.data_y)
        #     self.data_y = self.data_y[perm]

        # return self.data[int(iter_num % ro_num) * batch_size: int(iter_num% ro_num + 1) * batch_size] \
        #     , self.data_y[int(iter_num % ro_num) * batch_size: int(iter_num%ro_num + 1) * batch_size]



def normalization(matrix):
    matrix = matrix.copy()
    # amin, amax = -2000, 2000
    # try:
    amin, amax = matrix.min(),matrix.max()
    if amax-amin == 0:
        amin, amax = -2000, 2000
        matrix = (matrix-amin)/(amax-amin) - 0.5
    else:
        matrix = (matrix-amin)/(amax-amin) - 0.5
# except:
    
    return matrix

def truncate_hu_bone(image_array, max, min):
    image = image_array.copy()
    image[image > max] = max
    image[image < min] = min
    # image = normalization(image)
    return image

def truncate_hu(image_array, max, min):
    image = image_array.copy()
    image[image > max] = max
    image[image < min] = min
    image = normalization(image)
    return image

# def normalization2(image_array):
#     max = image_array.max()
#     min = image_array.min()
#     image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
#     avg = image_array.mean()
#     image_array = image_array-avg
#     return image_array   # a bug here, a array must be returned,directly appling function did't work

def one_shot_diam(diam):
    if diam < 10:
        return [1, 0, 0]
    if diam < 20:
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def one_shot_attri(attri):
    if attri <= 1:
        return [1, 0, 0, 0, 0]
    if attri <= 2:
        return [0, 1, 0, 0, 0]    
    if attri <= 3:
        return [0, 0, 1, 0, 0]    
    if attri <= 4:
        return [0, 0, 0, 1, 0]    
    else:
        return [0, 0, 0, 0, 1]

def get_image(image_path , is_grayscale = False):
    return np.array(inverse_transform(imread(image_path, is_grayscale)))


def save_images(images , size , image_path):
    return imsave(inverse_transform(images) , size , image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images , size , path):
    return scipy.misc.imsave(path , merge(images , size))

def merge(images , size):
    h , w = images.shape[1] , images.shape[2]
    img = np.zeros((h*size[0] , w*size[1] , 3))
    for idx , image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h +h , i*w : i*w+w , :] = image

    return img

def inverse_transform(image):
    return (image + 1.)/2.

def read_image_list(category):
    filenames = []
    print("list file")
    list = os.listdir(category)

    for file in list:
        filenames.append(category + "/" + file)

    print("list file ending!")

    return filenames

##from caffe
def vis_square(visu_path , data , type):
    """Take an array of shape (n, height, width) or (n, height, width , 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]) ,
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data , padding, mode='constant' , constant_values=1)  # pad with ones (white)

    # tilethe filters into an im age
    data = data.reshape((n , n) + data.shape[1:]).transpose((0 , 2 , 1 , 3) + tuple(range(4 , data.ndim + 1)))

    data = data.reshape((n * data.shape[1] , n * data.shape[3]) + data.shape[4:])

    plt.imshow(data[:,:,0])
    plt.axis('off')

    if type:
        plt.savefig('./{}/weights.png'.format(visu_path) , format='png')
    else:
        plt.savefig('./{}/activation.png'.format(visu_path) , format='png')


# def sample_label():
#     num = 64
#     label_vector = np.zeros((num , 13), dtype=np.float)
#     for i in range(0 , num):
#         label_vector[i , int(i/8)] = 1.0
#     return label_vector

def caseid_to_scanid(caseid):
    returnstr = ''
    if caseid < 10:
        returnstr = '000' + str(caseid)
    elif caseid < 100:
        returnstr = '00' + str(caseid)
    elif caseid < 1000:
        returnstr = '0' + str(caseid)
    else:
        returnstr = str(caseid)
    return 'LIDC-IDRI-' + returnstr


def sample_label():
    label_vector = []
    for i in range(8):

        label_vector.append(np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype = np.float))
        label_vector.append(np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype = np.float))
        label_vector.append(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype = np.float))
        label_vector.append(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype = np.float))
        label_vector.append(np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype = np.float))
        label_vector.append(np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype = np.float))
        label_vector.append(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype = np.float))
        label_vector.append(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype = np.float))

    return np.array(label_vector)


def sample_lungwindow():
    return np.zeros((64, 128, 128, 1), np.float)

def sample_mediastinumwindow():
    return np.zeros((64, 128, 128, 1), np.float)

def sample_masks():
    datapath = './testdata/'
    filelist = os.listdir(datapath)
    imagelist = []
    bone_mask_addrs = glob.glob('bone_chose/*.jpg')
    shuffle(bone_mask_addrs)
    bone_mask_addrs = bone_mask_addrs[:len(filelist)]

    for file,bone_addr in zip(filelist,bone_mask_addrs):
        temp = np.load(datapath + file)
        bone = cv2.imread(bone_addr,0)
        bone = bone / 255.
        temp = cv2.resize(temp, (128, 128)) + bone
        # cv2.imwrite('temp/'+str(file)+'.jpg',temp*255)
        for i in range(8):
            imagelist.append(temp)
    masks = np.array(imagelist)
    masks = np.expand_dims(masks, axis = 3)
    return masks

def sample_image():
    return np.zeros((64, 128, 128, 1), np.float)

def sample_masks_test():
    datapath = './masks/'
    filelist = os.listdir(datapath)
    shuffle(filelist)
    filelist = filelist[:8]
    imagelist = []
    bone_mask_addrs = glob.glob('bone_chose/*.jpg')
    shuffle(bone_mask_addrs)
    bone_mask_addrs = bone_mask_addrs[:len(filelist)]

    for file,bone_addr in zip(filelist,bone_mask_addrs):
        temp = np.load(datapath + file)
        bone = cv2.imread(bone_addr,0)
        bone = bone / 255.
        temp = cv2.resize(temp, (128, 128)) + bone
        # cv2.imwrite('temp/'+str(file)+'.jpg',temp*255)
        for i in range(8):
            imagelist.append(temp)
    masks = np.array(imagelist)
    masks = np.expand_dims(masks, axis = 3)
    return masks
