import os

import numpy as np
import nibabel as nib

def cal_subject_level_dice(prediction, target, class_num=2):# class_num是你分割的目标的类别个数
    '''
    step1: calculate the dice of each category
    step2: remove the dice of the empty category and background, and then calculate the mean of the remaining dices.
    :param prediction: the automated segmentation result, a numpy array with shape of (h, w, d)
    :param target: the ground truth mask, a numpy array with shape of (h, w, d)
    :param class_num: total number of categories
    :return:
    '''
    eps = 1e-10
    empty_value = -1.0
    dscs = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[i] = dsc
    dscs = np.where(dscs == -1.0, np.nan, dscs)
    subject_level_dice = np.nanmean(dscs[1:])
    return subject_level_dice

def evaluate_demo(prediction_nii_files, target_nii_files):
    '''
    This is a demo for calculating the mean dice of all subjects.
    :param prediction_nii_files: a list which contains the .nii file paths of predicted segmentation
    :param target_nii_files: a list which contains the .nii file paths of ground truth mask
    :return:
    '''
    dscs = []
    for prediction_nii_file, target_nii_file in zip(prediction_nii_files, target_nii_files):
        prediction_nii = nib.load(prediction_nii_file)
        prediction = prediction_nii.get_data()
        target_nii = nib.load(target_nii_file)
        target = target_nii.get_data()
        dsc = cal_subject_level_dice(prediction, target, class_num=2)
        dscs.append(dsc)
    return dsc
path1 = "/home/hjf/dataset1/Mutul_data/252_VP_test"
path2 = "/home/hjf/dataset1/Mutul_data/test_merge"
a = os.listdir(path1)
a.sort()
b = os.listdir(path2)
b.sort()
num = 0
for i in range(52):
    ab = evaluate_demo([os.path.join(path1, a[i])], [os.path.join(path2, b[i])])
    print(a[i], ":", ab)
    num += ab
print("average:", num/52)
#
# a =evaluate_demo(['/home/hjf/dataset1/shabi/CAO.nii.gz'],['/home/hjf/dataset1/Mutul_data/A250_merge/patientA028_merge.nii'])#前一个地址是你预测的三维图像的地址，后一个是标签地址
# print(a)
#
# path1 = '/home/hjf/daima/MAML-main/nnUNet_raw_data_base/nnUNet_raw_data/Task888_ces/labelsTr'
# path2 = '/home/hjf/daima/MAML-main/nnUNet_raw_data_base/nnUNet_preprocessed/Task888_ces/gt_segmentations'
# # s = os.listdir(path1)
# # s.sort()
# # a = s[:244]
# # b = s[244:]
# a = os.listdir(path1)
# a.sort()
# b = os.listdir(path2)
# b.sort()
# num = 0
# for i in range(244):
#     ab = evaluate_demo([os.path.join(path1, a[i])], [os.path.join(path1, b[i])])
#     print(a[i], ":", ab)
#     num += ab
# print("average:", num/244)