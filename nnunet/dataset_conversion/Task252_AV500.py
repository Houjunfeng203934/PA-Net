from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import numpy as np
from nnunet.configuration import default_num_threads
from scipy.ndimage import label


# def export_segmentations(indir, outdir):
#     niftis = subfiles(indir, suffix='nii.gz', join=False)
#     for n in niftis:
#         identifier = str(n.split("_")[-1][:-7])
#         outfname = join(outdir, "segmentation-%s.nii" % identifier)
#         img = sitk.ReadImage(join(indir, n))
#         sitk.WriteImage(img, outfname)
#
#
# def export_segmentations_postprocess(indir, outdir):
#     maybe_mkdir_p(outdir)
#     niftis = subfiles(indir, suffix='nii.gz', join=False)
#     for n in niftis:
#         print("\n", n)
#         identifier = str(n.split("_")[-1][:-7])
#         outfname = join(outdir, "segmentation-%s.nii" % identifier)
#         img = sitk.ReadImage(join(indir, n))
#         img_npy = sitk.GetArrayFromImage(img)
#         lmap, num_objects = label((img_npy > 0).astype(int))
#         sizes = []
#         for o in range(1, num_objects + 1):
#             sizes.append((lmap == o).sum())
#         mx = np.argmax(sizes) + 1
#         print(sizes)
#         img_npy[lmap != mx] = 0
#         img_new = sitk.GetImageFromArray(img_npy)
#         img_new.CopyInformation(img)
#         sitk.WriteImage(img_new, outfname)


if __name__ == "__main__":

    train_dir = "/home/hjf/dataset1/Mutul_data/A-500"
    train_dir1 = "/home/hjf/dataset1/Mutul_data/A-500mask"
    test_dir = "/home/hjf/dataset1/Mutul_data/testv"


    output_folder = "/home/hjf/daima/MAML-main/nnUNet_raw_data_base/nnUNet_raw_data/Task252_VA500"
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)


    def load_save_train(args):
        data_file, seg_file = args
        pat_id = data_file.split("/")[-1]
        pat_id = "train_" + pat_id.split("-")[-1][:-4]

        img_itk = sitk.ReadImage(data_file)
        sitk.WriteImage(img_itk, join(img_dir, pat_id + "_0000.nii.gz"))

        img_itk = sitk.ReadImage(seg_file)
        sitk.WriteImage(img_itk, join(lab_dir, pat_id + ".nii.gz"))
        return pat_id

    def load_save_test(args):
        data_file = args
        pat_id = data_file.split("/")[-1]
        pat_id = "test_" + pat_id.split("-")[-1][:-4]

        img_itk = sitk.ReadImage(data_file)
        sitk.WriteImage(img_itk, join(img_dir_te, pat_id + "_0000.nii.gz"))
        return pat_id

    nii_files_tr_data = subfiles(train_dir, True, "patient", "nii", True) # 获得train_dir中包含volume,且以nii结尾的数据
    nii_files_tr_seg = subfiles(train_dir1, True, "patient", "nii", True) # 获得train_dir中包含segmen，且以nii结尾的数据
    nii_files_ts = subfiles(test_dir, True, "patient", "nii", True) # 获得test_dir中包含volume，且以nii结尾的数据
    p = Pool(default_num_threads)
    train_ids = p.map(load_save_train, zip(nii_files_tr_data, nii_files_tr_seg))
    test_ids = p.map(load_save_test, nii_files_ts)
    p.close()
    p.join()

    # train_ids = []
    # for i in zip(nii_files_tr_data, nii_files_tr_seg):
    #     train_ids.append(load_save_train(i))
    # test_ids = []
    # for i in nii_files_ts:
    #     test_ids.append(load_save_test(i))


    json_dict = OrderedDict()
    json_dict['name'] = "LITS"
    json_dict['description'] = "LITS"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "tumor"
    }

    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)