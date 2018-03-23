import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset


class SlamDataset(Dataset):
    def __init__(self, fname, dset_name, flatten = True):
        self.f = h5.File(fname, 'r')


        tmp = np.array(self.f[dset_name + "/rgb"], dtype=np.float32)
        if flatten:
            tmp = tmp.reshape((tmp.shape[0], -1))
        else:
            tmp = np.transpose(tmp, [0, 3, 1, 2])
        self.rgb = torch.from_numpy(tmp)

        tmp = np.array(self.f[dset_name + "/depth"], dtype=np.float32)
        if flatten:
            tmp = tmp.reshape((tmp.shape[0], -1))
        else:
            tmp = np.reshape(tmp, (tmp.shape[0], 1, tmp.shape[1], tmp.shape[2]))
        self.depth = torch.from_numpy(tmp)

        # h5py doesn't seem to like it when we add a 'dtype=np.float32' to the
        # np.array call. Doing the conversion in two steps produces the result
        tmp = np.array(self.f[dset_name + "/pose"])
        self.pose = torch.from_numpy(tmp.astype(dtype=np.float32))


    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        sample = {'rgb' : self.rgb[idx], 'depth' : self.depth[idx], 'pose' : self.pose[idx]}
        return sample
