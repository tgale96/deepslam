import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset

# TODO(tgale): If this constructor takes too long, build
# rgb_pair and no_change quat into hdf5 dataset
class SlamDataset(Dataset):
    def __init__(self, fname, dset_name, flatten = True, pair_data = False):
        self.f = h5.File(fname, 'r')

        tmp = np.array(self.f[dset_name + "/rgb"], dtype=np.float32)
        if flatten:
            tmp = tmp.reshape((tmp.shape[0], -1))
        else:
            tmp = np.transpose(tmp, [0, 3, 1, 2])
        self.rgb = torch.from_numpy(tmp)
        print("rgb shape: {}".format(self.rgb.shape))

        # Create dataset of images with their previous image
        # Image 0 gets itself, and pose change will be identity
        self.pair_data = pair_data
        if self.pair_data:
            rgb_pair = []
            for i in range(tmp.shape[0]):
                prev = max(i-1, 0)
                a = np.expand_dims(tmp[i, :], axis=0)
                b = np.expand_dims(tmp[prev, :], axis=0)
                rgb_pair.append(np.concatenate((a, b)))
            rgb_pair = np.array(rgb_pair)

            self.rgb_pair = torch.from_numpy(rgb_pair)
            print("rgb pair shape: {}".format(self.rgb_pair.shape))

        tmp = np.array(self.f[dset_name + "/depth"], dtype=np.float32)
        if flatten:
            tmp = tmp.reshape((tmp.shape[0], -1))
        else:
            tmp = np.reshape(tmp, (tmp.shape[0], 1, tmp.shape[1], tmp.shape[2]))
        tmp = tmp / 5000.
        self.depth = torch.from_numpy(tmp)
        print("depth shape: {}".format(self.depth.shape))

        # h5py doesn't seem to like it when we add a 'dtype=np.float32' to the
        # np.array call. Doing the conversion in two steps produces the result
        tmp = np.array(self.f[dset_name + "/pose"])
        tmp = tmp.astype(dtype=np.float32)
        self.pose = torch.from_numpy(tmp)
        
        if self.pair_data:
            # Add the no_change quaternion for the first pose_diff
            tmp = np.array(self.f[dset_name + "/pose_diff"])
            tmp = tmp.astype(dtype=np.float32)
            no_change = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
            tmp = np.concatenate((no_change, tmp))
            self.pose_diff = torch.from_numpy(tmp)
            print("pose diff shape: {}".format(self.pose_diff.shape))

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        if self.pair_data:
            sample = {'rgb' : self.rgb[idx], 
                      'depth' : self.depth[idx], 
                      'pose' : self.pose[idx]}
        else:
            sample = {'rgb_pair' : self.rgb_pair[idx], 
                      'depth' : self.depth[idx], 
                      'pose_diff' : self.pose_diff[idx]}
        return sample
