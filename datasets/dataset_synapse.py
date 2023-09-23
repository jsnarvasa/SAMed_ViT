import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic


def random_rot_flip(image, label):
    if len(image.shape) == 3:
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2))
        label = np.rot90(label, k)
        axis = np.random.randint(1, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis-1).copy()
        return image, label
    elif len(image.shape) == 4:
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(2, 3))
        label = np.rot90(label, k)
        axis = np.random.randint(2, 4)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis-2).copy()
        return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label, doy = sample['image'], sample['label'], sample['doy']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        if len(image.shape) == 3:
            t, x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label_h, label_w = label.shape
            low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(1)
            image = repeat(image, 't c h w -> t (repeat c) h w', repeat=3)
            label = torch.from_numpy(label.astype(np.float32))
            low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        elif len(image.shape) == 4:
            t, c, x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label_h, label_w = label.shape
            low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
            low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long(), 'doy': doy}
        return sample
    

class Pad_Timeseries(object):
    def __init__(self, dtype=torch.float32):

        self.dtype = dtype
        return
    

    def __call__(self, sample):
        image = sample['image']
        doy = sample['doy']

        # Get the current length of timeseries observation
        timeseries_length = image.shape[0]

        diff = 60 - timeseries_length

        if diff > 0:
            # creates the shape of the padding that we need [diff, channel, height, width]
            pad_shape = [diff] + list(image.shape)[1:]
            image = torch.cat((image, torch.zeros(pad_shape, dtype=self.dtype)), dim = 0)
            doy_pad_shape = [diff] + list(doy.shape)[1:]
            doy = torch.cat((doy, torch.zeros(doy_pad_shape)), dim=0)
        elif diff < 0:
            # if for some reason, we have an instance where the number of timeseries observations is greater than 60
            # we will just take the first 60 observations
            image = image[:60, ...]
            doy = doy[:60, ...]
        
        sample['image'] = image
        sample['doy'] = doy
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label, doy = data['image'], data['label'], data['doy']
        elif self.split == "pastis_test_vol":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label, doy = data['image'], data['label'].astype('uint8'), data['doy']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        H, W = label.shape
        doy_tensor_list = [torch.full((H, W), value) for value in doy]
        doy = torch.stack(doy_tensor_list, dim=0)

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label, 'doy': doy}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        
        return sample
