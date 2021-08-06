import torch.utils.data as data
import os.path
import csv
from data.utils import get_params, get_transform, make_dataset
from PIL import Image
import numpy as np
from skimage.transform import resize
import torch
import torch.nn.functional as F
import torchvision.transforms as T

class SevirDataset(data.Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        # save opt to avoid mutability bug
        self.fraction_dataset = opt.fraction_dataset
        self.phase = opt.phase
        # folder setup
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, self.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # get image paths
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        # z-score normalization
        znorm_mu_sevir = [ 1.38836827e+03, -3.64035706e+03, -1.43816570e+03, 1.88870760e+01, 3.09813984e-02]
        znorm_sigma_sevir = [2.01801387e+03, 1.17885038e+03, 2.59019927e+03, 3.66931761e+01, 5.13092746e-01]
        self.znorm = dict()
        ind_to_type = {0: 'vis', 1: 'ir069', 2: 'ir107', 3: 'vil', 4: 'lght'}
        for key, value in ind_to_type.items():
            self.znorm[value] = (znorm_mu_sevir[key], znorm_sigma_sevir[key])

    def resize_numpy(self, matrix, target_size):
        number_frames, number_channels, w, h = matrix.shape
        matrix = matrix.reshape((number_frames * number_channels, w, h))
        matrix = [resize(x, (target_size, target_size), preserve_range=True) for x in matrix]
        matrix = np.array(matrix).reshape(number_frames, number_channels, target_size, target_size)
        return matrix

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = np.load(AB_path).astype(np.float32)
        ind_to_type = {0: 'vis', 1: 'ir069', 2: 'ir107', 3: 'vil', 4: 'lght'}
        # choose number of frames
        n_samples_per_task = self.opt.n_support + self.opt.n_query
        AB = torch.from_numpy(AB[:n_samples_per_task, :, :, :])
        # normalize
        for i in [1,2,3,4]: #don't use vis at all
            mu, sigma  = self.znorm[ind_to_type[i]]
            AB[:, i] = (AB[:, i] - mu) / sigma
        # separate source and target
        A, B = AB[:, (1,2,4), :, :], AB[:, 3:4, :, :] # split AB image into A and B
        # resize A to opt.load_size
        A = F.interpolate(A, size=self.opt.load_size)
        # resize B
        target_size = self.opt.target_size if self.opt.resize_target else self.opt.load_size
        B =  F.interpolate(B, size=target_size)
        # done
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return int(len(self.AB_paths) / self.fraction_dataset)


