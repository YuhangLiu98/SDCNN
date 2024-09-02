import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import time
import os
import scipy.io
from scipy.ndimage.interpolation import rotate
from multiprocessing import Pool
from glob import glob  # glob模块用来查找文件目录和文件，并将搜索的到的结果返回到一个列表中，
import copy


def process_image(train_noisy):
    STD_train = []
    for h in range(3, train_noisy.shape[1] - 3):
        for w in range(3, train_noisy.shape[2] - 3):
            STD_train.append(
                np.std(train_noisy[:, h - 3:h + 3, w - 3:w + 3, :].reshape([-1, 36, 1]), 1).reshape([-1, 1, 1]))
    return np.mean(np.concatenate(STD_train, 1), 1)


def horizontal_flip(image, rate=0.5):
    image = image[:, ::-1, :]
    return image


def vertical_flip(image, rate=0.5):
    image = image[::-1, :, :]
    return image


def random_rotation(image, angle):
    h, w, _ = image.shape
    image = rotate(image, angle)
    return image


class benchmark_data(Dataset):

    def __init__(self, data_dir, task, transform=None):

        self.task = task
        self.data_dir = data_dir
        input_path = sorted(glob(os.path.join(self.data_dir, '*_input.npy')))
        target_path = sorted(glob(os.path.join(self.data_dir, '*_target.npy')))
        normal_path = copy.deepcopy(target_path)
        random.shuffle(normal_path)
        self.input_ = [np.load(f) for f in input_path]
        self.target_ = [np.load(f) for f in target_path]
        self.normal_ = [np.load(f) for f in normal_path]
        self.input_ = np.array(self.input_)
        self.target_ = np.array(self.target_)
        self.normal_ = np.array(self.normal_)
        self.input_ = self.input_.reshape([-1, 512, 512, 1])
        self.target_ = self.target_.reshape([-1, 512, 512, 1])
        self.normal_ = self.normal_.reshape([-1, 512, 512, 1])
        self.data_num = self.input_.shape[0]

        self.indices = self._indices_generator()
        self.patch_size = 128

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):

        def data_loader():

            if self.task == "test":
                Img_noisy, Img_GT, Img_Normal = self.input_[index], self.target_[index], self.normal_[index]
                Img_noisy = (np.transpose(Img_noisy, (2, 0, 1)))
                Img_GT = (np.transpose(Img_GT, (2, 0, 1)))
                Img_Normal = (np.transpose(Img_Normal, (2, 0, 1)))

            if self.task == "train":
                Img_noisy, Img_GT, Img_Normal = self.input_[index], self.target_[index], self.normal_[index]

                # Augmentation
                horizontal = torch.randint(0, 2, (1,))
                vertical = torch.randint(0, 2, (1,))
                rand_rot = torch.randint(0, 4, (1,))
                rot = [0, 90, 180, 270]
                if horizontal == 1:
                    Img_noisy = horizontal_flip(Img_noisy)
                    Img_GT = horizontal_flip(Img_GT)
                    Img_Normal = horizontal_flip(Img_Normal)
                if vertical == 1:
                    Img_noisy = vertical_flip(Img_noisy)
                    Img_GT = vertical_flip(Img_GT)
                    Img_Normal = vertical_flip(Img_Normal)
                Img_noisy = random_rotation(Img_noisy, rot[rand_rot])
                Img_GT = random_rotation(Img_GT, rot[rand_rot])
                Img_Normal = random_rotation(Img_Normal, rot[rand_rot])

                Img_noisy = (np.transpose(Img_noisy, (2, 0, 1)))
                Img_GT = (np.transpose(Img_GT, (2, 0, 1)))
                Img_Normal = (np.transpose(Img_Normal, (2, 0, 1)))
                x_00 = torch.randint(0, Img_noisy.shape[1] - self.patch_size, (1,))
                y_00 = torch.randint(0, Img_noisy.shape[2] - self.patch_size, (1,))
                Img_noisy = Img_noisy[:, x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]
                Img_GT = Img_GT[:, x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]
                Img_Normal = Img_Normal[:, x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]

            return np.array(Img_noisy, dtype=np.float32), np.array(Img_GT, dtype=np.float32), np.array(Img_Normal,
                                                                                                       dtype=np.float32), index  # ,Img_train, Img_train_noisy

        def _timeprint(isprint, name, prevtime):
            if isprint:
                print('loading {} takes {} secs'.format(name, time() - prevtime))
            return time()

        if torch.is_tensor(index):
            index = index.tolist()

        input_noisy, input_GT, input_normal, idx = data_loader()
        target = {
            'dir_idx': str(idx)
        }

        return target, input_noisy, input_GT, input_normal

    def _indices_generator(self):

        return np.arange(self.data_num, dtype=int)


if __name__ == "__main__":
    time_print = True

    prev = time()
