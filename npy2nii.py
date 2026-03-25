import SimpleITK as sitk
import os
import numpy as np
from tqdm import trange
from options.test_options import TestOptions


def get_npy_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.npy':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


if __name__ == '__main__':
    opt = TestOptions().parse()
    nii_path = os.path.join(opt.dataroot, 'test_nii')
    save_path = os.path.join(opt.results_dir, opt.name + 'nii')
    os.makedirs(save_path, exist_ok=True)
    patients_name_list = os.listdir(nii_path)
    patients_path_list = [os.path.join(nii_path, i) for i in patients_name_list]
    patients_model_y_list = [os.path.join(i, opt.model_y) for i in patients_path_list]
    patients_name_list.sort()
    patients_path_list.sort()
    patients_model_y_list.sort()

    for i in range(len(patients_model_y_list)):
        sitk_img = sitk.ReadImage(patients_model_y_list[i]+'.nii.gz')
        img_arr = sitk.GetArrayFromImage(sitk_img)
        img_list = [i for i in get_npy_listdir(os.path.join(opt.results_dir, opt.name, patients_name_list[i]))]
        img_list.sort()
        new_img = np.zeros_like(img_arr)
        for j in trange(len(img_list)):
            image = np.load(img_list[j])
            new_img[j, :, :] = image
        new_img = sitk.GetImageFromArray(new_img)
        new_img.SetDirection(sitk_img.GetDirection())
        new_img.SetSpacing(sitk_img.GetSpacing())
        new_img.SetOrigin(sitk_img.GetOrigin())
        sitk.WriteImage(new_img, os.path.join(save_path, patients_name_list[i] + '.nii.gz'))
