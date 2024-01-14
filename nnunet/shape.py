import SimpleITK as sitk

img = sitk.ReadImage('/nnUNet_raw_data_base/nnUNet_raw_data/Task229_Liver/imagesTr/train_49_0000.nii.gz')
img1 = sitk.GetArrayFromImage(img)
print(img1)
pa = sitk.ReadImage('/nnUNet_raw_data_base/nnUNet_raw_data/Task229_Liver/labelsTr/train_49.nii.gz')
pb = sitk.GetArrayFromImage(pa)
print(pb)