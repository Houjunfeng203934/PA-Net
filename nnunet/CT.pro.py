import SimpleITK as sitk
import numpy as np
import os


def saved_preprocessed(savedImg, origin, direction, xyz_thickness, saved_name):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    sitk.WriteImage(newImg, saved_name)


def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


ct_path = 'C:\\Users\\VULCAN\\Desktop\\LITS'
saved_path = 'C:\\Users\\VULCAN\\Desktop\\wl'
name_list = ['volume-1.nii']
for name in name_list:
    ct = sitk.ReadImage(os.path.join(ct_path, name))
    origin = ct.GetOrigin()
    direction = ct.GetDirection()
    xyz_thickness = ct.GetSpacing()
    ct_array = sitk.GetArrayFromImage(ct)
    seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ct_path, name.replace('volume', 'segmentation'))))
    seg_bg = seg_array == 0
    seg_liver = seg_array >= 1
    seg_tumor = seg_array == 2

    ct_bg = ct_array * seg_bg
    ct_liver = ct_array * seg_liver
    ct_tumor = ct_array * seg_tumor

    liver_min = ct_liver.min()
    liver_max = ct_liver.max()
    tumor_min = ct_tumor.min()
    tumor_max = ct_tumor.max()

    # by liver
    liver_wide = liver_max - liver_min
    liver_center = (liver_max + liver_min) / 2
    liver_wl = window_transform(ct_array, liver_wide, liver_center, normal=True)
    saved_name = os.path.join(saved_path, 'liver_wl_1.nii')
    saved_preprocessed(liver_wl, origin, direction, xyz_thickness, saved_name)

    # by tumor (recommended)
    tumor_wide = tumor_max - tumor_min
    tumor_center = (tumor_max + tumor_min) / 2
    tumor_wl = window_transform(ct_array, tumor_wide, tumor_center, normal=True)
    saved_name = os.path.join(saved_path, 'tumor_wl_1.nii')
    saved_preprocessed(tumor_wl, origin, direction, xyz_thickness, saved_name)
