import os.path
import numpy as np
from torch.utils import data
import torch


class SevirDataset(data.Dataset):
    def __init__(self,
                 dataroot,
                 phase,
                 load_size,
                 crop_size,
                 frames_per_event,
                 fraction_dataset,
                 tgt_size,
                 transform=None):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.transform = transform

        self.root = dataroot
        self.load_size = load_size
        self.crop_size = crop_size
        self.frames_per_event = frames_per_event
        self.fraction_dataset = fraction_dataset

        # filestructure
        self.dir_AB = os.path.join(dataroot, phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # get image paths

        # sevir norm constants
        self.znorm_mu_sevir = [
            1.38836827e+03, -3.64035706e+03, -1.43816570e+03, 1.88870760e+01, 3.09813984e-02]
        self.znorm_sigma_sevir = [
            2.01801387e+03, 1.17885038e+03, 2.59019927e+03, 3.66931761e+01, 5.13092746e-01]
        self.ind_to_type = {0: 'vis', 1: 'ir069',
                            2: 'ir107', 3: 'vil', 4: 'lght'}

        # input and output have different dimensions
        self.tgt_size = tgt_size
        # crop_size should be smaller than the size of loaded image
        assert (self.load_size >= self.crop_size)

        self.znorm = dict()
        for key, value in self.ind_to_type.items():
            self.znorm[value] = (self.znorm_mu_sevir[key],
                                 self.znorm_sigma_sevir[key])

        self.even_ids = np.arange(0, 48, step=2)
        self.odd_ids = np.arange(1, 48, step=2)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        """
        # read an event given a random integer index
        AB_path = self.AB_paths[index]
        AB = np.load(AB_path)
        # 48x5x384x384 (ignore the last frame, and VIS and VIL)
        views = torch.FloatTensor(AB[:-1, [1, 2, 4], :, :])
        query = torch.stack([self.transform(views[i]) for i in self.even_ids])
        key = torch.stack([self.transform(views[i]) for i in self.odd_ids])
        return [query, key]

    def __len__(self):
        """Return the total number of images in the dataset."""
        return int(len(self.AB_paths) / self.fraction_dataset)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', ".npy"
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SEVIR dataset options')
    parser.add_argument('--dataroot', type=str,
                        default='/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/image_translation/')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--load_size', type=int, default=192)
    parser.add_argument('--crop_size', type=int, default=192)
    parser.add_argument('--tgt_size', type=int, default=384)
    parser.add_argument('--fraction_dataset', type=int, default=1)
    parser.add_argument('--frames_per_event', type=int, default=49)
    opt = parser.parse_args()

    # opt to dict
    opt = vars(opt)

    data = SevirDataset(**opt)
    print(len(data))
