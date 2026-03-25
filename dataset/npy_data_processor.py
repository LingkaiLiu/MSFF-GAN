import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchio as tio


class ImageDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        target_dir = os.path.join(opt.dataroot, opt.phase)
        patients_name_list = os.listdir(target_dir)
        self.model_y = opt.model_y
        self.model_x_list = opt.model_xs
        if isinstance(self.model_x_list, str):
            self.model_x_list = [opt.model_xs]

        self.all_image_list = []

        for patient in patients_name_list:
            patient_model_y = os.path.join(target_dir, patient, self.model_y)
            for slicer in os.listdir(patient_model_y):
                self.all_image_list.append(os.path.join(target_dir, patient, self.model_y, slicer))

        # check files count across models
        data_x_nii_files_count_list = np.array(
            [len(os.listdir(os.path.join(target_dir, patient))) for patient in patients_name_list])
        print(data_x_nii_files_count_list)
        print(f'NPY files\'s count checked, {len(data_x_nii_files_count_list)} files in each model.')

    def __getitem__(self, index):
        name = self.all_image_list[index]
        data_return = {}
        A = np.array([np.load(name.replace(self.model_y, model))for model in self.model_x_list])
        B = np.expand_dims(np.load(name).astype(np.float32), axis=0)

        data_return['A'] = torch.from_numpy(A).type(torch.FloatTensor)
        data_return['B'] = torch.from_numpy(B).type(torch.FloatTensor)

        Resize_transform_A = tio.transforms.Resize(
            target_shape=(data_return['A'].shape[0], self.opt.fineSize, self.opt.fineSize),
            image_interpolation='linear')
        Resize_transform_B = tio.transforms.Resize(
            target_shape=(data_return['B'].shape[0], self.opt.fineSize, self.opt.fineSize),
            image_interpolation='linear')
        data_return['A'] = Resize_transform_A(data_return['A'].unsqueeze(dim=0)).squeeze(dim=0)
        data_return['B'] = Resize_transform_B(data_return['B'].unsqueeze(dim=0)).squeeze(dim=0)

        data_return['A_paths'] = name
        data_return['B_paths'] = name

        return data_return

    def __len__(self):
        return len(self.all_image_list)


class ImageDatasetTest(Dataset):
    def __init__(self, opt):
        self.opt = opt
        target_dir = os.path.join(opt.dataroot, opt.phase)
        patients_name_list = os.listdir(target_dir)
        self.model_y = opt.model_y
        self.model_x_list = opt.model_xs
        if isinstance(self.model_x_list, str):
            self.model_x_list = [opt.model_xs]

        self.all_image_list = []

        for patient in patients_name_list:
            patient_model_y = os.path.join(target_dir, patient, self.model_y)
            for slicer in os.listdir(patient_model_y):
                self.all_image_list.append(os.path.join(target_dir, patient, self.model_y, slicer))

        # check files count across models
        data_x_nii_files_count_list = np.array(
            [len(os.listdir(os.path.join(target_dir, patient))) for patient in patients_name_list])
        print(data_x_nii_files_count_list)
        print(f'NPY files\'s count checked, {len(data_x_nii_files_count_list)} files in each model.')

    def __getitem__(self, index):
        name = self.all_image_list[index]
        data_return = {}
        A = np.array([np.load(name.replace(self.model_y, model)) for model in self.model_x_list])
        B = np.expand_dims(np.load(name), axis=0)

        data_return['A'] = torch.from_numpy(A).type(torch.FloatTensor)
        data_return['B'] = torch.from_numpy(B).type(torch.FloatTensor)

        Resize_transform_A = tio.transforms.Resize(
            target_shape=(data_return['A'].shape[0], self.opt.fineSize, self.opt.fineSize),
            image_interpolation='linear')
        Resize_transform_B = tio.transforms.Resize(
            target_shape=(data_return['B'].shape[0], self.opt.fineSize, self.opt.fineSize),
            image_interpolation='linear')
        data_return['A'] = Resize_transform_A(data_return['A'].unsqueeze(dim=0)).squeeze(dim=0)
        data_return['B'] = Resize_transform_B(data_return['B'].unsqueeze(dim=0)).squeeze(dim=0)

        data_return['A_paths'] = name
        data_return['B_paths'] = name
        data_return['A_origin_shape'] = torch.from_numpy(A).shape
        return data_return

    def __len__(self):
        return len(self.all_image_list)
