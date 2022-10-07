# -*- coding: UTF-8 -*-


import glob
import numpy as np
import pandas as pd
from PIL import Image
import random

# h,w = 60,50
h, w = (60, 50)
size = h * w
# Receding_Hairline Wearing_Necktie Rosy_Cheeks Eyeglasses  Goatee  Chubby
#  Sideburns   Blurry  Wearing_Hat Double_Chin Pale_Skin   Gray_Hair   Mustache    Bald
label_cls = 'Eyeglasses'

pngs = sorted(glob.glob('./data/img_align_celeba/*.jpg'))
data = pd.read_table('./data/list_attr_celeba.txt',
                     delim_whitespace=True, error_bad_lines=False)

eyeglasses = np.array(data[label_cls])
eyeglasses_cls = (eyeglasses + 1)/2

label_glasses = np.zeros((202599, 2))
correct_list = []
correct_list_test = []
false_list = []
false_list_test = []
for i in range(len(label_glasses)):
    if eyeglasses_cls[i] == 1:
        label_glasses[i][1] = 1
        if i < 160000:
            correct_list.append(i)
        else:
            correct_list_test.append(i)
    else:
        label_glasses[i][0] = 1
        if i < 160000:
            false_list.append(i)
        else:
            false_list_test.append(i)

print(len(correct_list_test), len(false_list_test))

training_set_label = label_glasses[0:160000, :]
test_set_label = label_glasses[160000:, :]
training_set_cls = eyeglasses_cls[0:160000]
test_set_cls = eyeglasses_cls[160000:]


def create_trainbatch(num=10, channel=0):
    train_num = random.sample(false_list, num)
    if channel == 0:
        train_set = np.zeros((num, h, w))
    else:
        train_set = np.zeros((num, h, w, 3))

    train_set_label_ = []
    train_set_cls_ = []

    for i in range(num):
        img = Image.open(pngs[train_num[i]])
        img_grey = img.resize((w, h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            train_set[i, :, :] = img_grey
        else:
            img_grey = np.array(img_grey)
            train_set[i, :, :, :] = img_grey

        train_set_label_.append(training_set_label[train_num[i]])
        train_set_cls_.append(training_set_cls[train_num[i]])
    # if channel == 0:
    #     train_set = train_set.reshape(size,num).T
    train_set_label_new = np.array(train_set_label_)
    train_set_cls_new = np.array(train_set_cls_)

    return train_set/255, train_set_label_new, train_set_cls_new


def create_trainbatch_all_correct(num=10, channel=0):

    train_num = random.sample(correct_list, num)
    if channel == 0:
        train_set = np.zeros((num, h, w))
    else:
        train_set = np.zeros((num, h, w, 3))
    train_set_label_ = []
    train_set_cls_ = []
    n = 0
    for i in range(num):
        img = Image.open(pngs[train_num[i]])
        img_grey = img.resize((w, h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            train_set[i, :, :] = img_grey
        else:
            img_grey = np.array(img_grey)
            train_set[i, :, :, :] = img_grey

        train_set_label_.append(training_set_label[train_num[i]])
        train_set_cls_.append(training_set_cls[train_num[i]])
    # if channel == 0:
    #     train_set = train_set.reshape(size,num).T
    train_set_label_new = np.array(train_set_label_)
    train_set_cls_new = np.array(train_set_cls_)

    return train_set/255, train_set_label_new, train_set_cls_new


def create_trainbatch_(num=10, channel=0):
    train_num1 = random.sample(correct_list, int(num/2))
    train_num2 = random.sample(false_list, int(num/2))

    train_num = train_num1+train_num2
    if channel == 0:
        train_set = np.zeros((num, h, w))
    else:
        train_set = np.zeros((num, h, w, 3))
    train_set_label_ = []
    train_set_cls_ = []
    n = 0
    for i in range(num):
        img = Image.open(pngs[train_num[i]])
        img_grey = img.resize((w, h))
        if channel == 0:

            img_grey = np.array(img_grey.convert('L'))
            train_set[i, :, :] = img_grey
        else:
            img_grey = np.array(img_grey)
            train_set[i, :, :, :] = img_grey

        train_set_label_.append(training_set_label[train_num[i]])
        train_set_cls_.append(training_set_cls[train_num[i]])
    # if channel == 0:
    #     train_set = train_set.reshape(size,num).T
    train_set_label_new = np.array(train_set_label_)
    train_set_cls_new = np.array(train_set_cls_)

    return train_set/255, train_set_label_new, train_set_cls_new


def create_trainbatch_grad(num=200, channel=0):
    train_num1 = random.sample(correct_list, int(10))
    train_num2 = random.sample(false_list, int(190))

    train_num = train_num1+train_num2

    if channel == 0:
        train_set = np.zeros((num, h, w))
    else:
        train_set = np.zeros((num, h, w, 3))

    train_set_label_ = []
    train_set_cls_ = []
    n = 0
    for i in range(num):
        img = Image.open(pngs[train_num[i]])
        img_grey = img.resize((w, h))
        if channel == 0:

            img_grey = np.array(img_grey.convert('L'))
            train_set[i, :, :] = img_grey
        else:
            img_grey = np.array(img_grey)
            train_set[i, :, :, :] = img_grey

        train_set_label_.append(training_set_label[train_num[i]])
        train_set_cls_.append(training_set_cls[train_num[i]])
    # if channel == 0:
    #     train_set = train_set.reshape(size,num).T
    train_set_label_new = np.array(train_set_label_)
    train_set_cls_new = np.array(train_set_cls_)

    return train_set/255, train_set_label_new, train_set_cls_new


def create_testset(num=100, channel=0):

    test_num1 = random.sample(correct_list_test, num)
    test_num2 = random.sample(false_list_test, num)
    test_num = test_num1 + test_num2
    if channel == 0:
        test_set = np.zeros((num*2, h, w))
    else:
        test_set = np.zeros((num*2, h, w, 3))

    test_set_label_ = []
    test_set_cls_ = []

    for i in range(num*2):
        img = Image.open(pngs[test_num[i]])
        img_grey = img.resize((w, h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            test_set[i, :, :] = img_grey
        else:
            img_grey = np.array(img_grey)
            test_set[i, :, :, :] = img_grey

        test_set_label_.append(label_glasses[test_num[i]])
        test_set_cls_.append(eyeglasses_cls[test_num[i]])
    # if channel == 0:
    #     test_set = test_set.reshape(size,num*2).T
    test_set_label_new = np.array(test_set_label_)
    test_set_cls_new = np.array(test_set_cls_)

    return test_set/255, test_set_label_new, test_set_cls_new, test_set_cls_new.mean()*100


def create_testset_all(channel=0):

    test_num1 = random.sample(correct_list_test, len(correct_list_test))
    test_num2 = random.sample(false_list_test, len(false_list_test))
    test_num = test_num1 + test_num2
    # test_num =
    num = len(test_num)
    if channel == 0:
        test_set = np.zeros((num, h, w))
    else:
        test_set = np.zeros((num, h, w, 3))

    test_set_label_ = []
    test_set_cls_ = []

    for i in range(num):
        img = Image.open(pngs[test_num[i]])
        img_grey = img.resize((w, h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            test_set[i, :, :] = img_grey
        else:
            img_grey = np.array(img_grey)
            test_set[i, :, :, :] = img_grey

        test_set_label_.append(label_glasses[test_num[i]])
        test_set_cls_.append(eyeglasses_cls[test_num[i]])
    # if channel == 0:
    #     test_set = test_set.reshape(size,num).T
    test_set_label_new = np.array(test_set_label_)
    test_set_cls_new = np.array(test_set_cls_)

    return test_set/255, test_set_label_new, test_set_cls_new, test_set_cls_new.mean()*100


def create_testset_unbalanced(channel=0):

    test_num1 = random.sample(correct_list_test, 10)
    test_num2 = random.sample(false_list_test, 190)
    test_num = test_num1 + test_num2
    # test_num =
    num = len(test_num)
    if channel == 0:
        test_set = np.zeros((num, h, w))
    else:
        test_set = np.zeros((num, h, w, 3))

    test_set_label_ = []
    test_set_cls_ = []

    for i in range(num):
        img = Image.open(pngs[test_num[i]])
        img_grey = img.resize((w, h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            test_set[i, :, :] = img_grey
        else:
            img_grey = np.array(img_grey)
            test_set[i, :, :, :] = img_grey

        test_set_label_.append(label_glasses[test_num[i]])
        test_set_cls_.append(eyeglasses_cls[test_num[i]])
    # if channel == 0:
    #     test_set = test_set.reshape(size,num).T
    test_set_label_new = np.array(test_set_label_)
    test_set_cls_new = np.array(test_set_cls_)

    return test_set/255, test_set_label_new, test_set_cls_new, test_set_cls_new.mean()*100
