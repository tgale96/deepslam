import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset


class SlamDataset(Dataset):
    def __init__(self, fname, dset_name):
        self.f = h5.File(fname, 'r')

        # Flatten the image dims, for testing with fc network
        tmp = np.array(self.f[dset_name + "/rgb"], dtype=np.float32)
        tmp = tmp.reshape((tmp.shape[0], -1))
        self.rgb = torch.from_numpy(tmp)

        self.depth = torch.from_numpy(np.array(self.f[dset_name + "/depth"], dtype=np.float32))

        # h5py doesn't seem to like it when we add a 'dtype=np.float32' to the
        # np.array call. Doing the conversion in two steps produces the result
        tmp = np.array(self.f[dset_name + "/pose"])
        self.pose = torch.from_numpy(tmp.astype(dtype=np.float32))


    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        sample = {'rgb' : self.rgb[idx], 'depth' : self.depth[idx], 'pose' : self.pose[idx]}
        return sample
