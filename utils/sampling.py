import numpy as np
from torch.utils.data import Dataset
import cv2

class cusDataset(Dataset):
    def __init__(self, csidata, imagedata, jointsdata):
        self.csidata = csidata
        self.imagedata = imagedata
        self.jointsdata = jointsdata

    def __len__(self):
        return self.csidata.shape[3]

    def __getitem__(self, idx):
        csi = self.csidata[:,:,:,idx]
        joint = self.jointsdata[:, :, idx]
        image = self.imagedata[:,:,idx]
        # pdb.set_trace()
        # image = cv2.resize(image, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, dsize=(114, 114), interpolation=cv2.INTER_CUBIC)
        
        return [joint, image], csi


def split_dataset(csi_dataset, joints_dataset, image_dataset, train_ratio=0.9, num_ges=8, num_perges=1000):
    """
    Split the dataset into train and test sets with each label evenly distributed.

    Parameters:
    - dataset: numpy array of shape [:, :, :, num_samples]
    - labels: numpy array of shape (num_samples,)
    - train_ratio: the ratio of the dataset to be used for training (default is 0.9)

    Returns:
    - train_data: numpy array of shape [:, :, :, num_train_samples]
    - test_data: numpy array of shape [:, :, :, num_test_samples]
    - train_labels: numpy array of shape (num_train_samples,)
    - test_labels: numpy array of shape (num_test_samples,)
    """

    train_csi, test_csi, train_joints, test_joints, train_image, test_image = [], [], [], [], [], []

    for i in range(num_ges):
        # Select indices for the current label
        indices = list(range(i*num_perges, (i+1)*num_perges))
        num_samples = len(indices)

        # Calculate the number of samples for training
        num_train_samples = int(train_ratio * num_samples)

        # Randomly shuffle indices
        # np.random.shuffle(indices)

        # Split indices into train and test
        train_indices = indices[:num_train_samples]
        test_indices = indices[num_train_samples:]

        # Append data and labels to train and test sets
        train_csi.append(csi_dataset[:, :, :, train_indices])
        test_csi.append(csi_dataset[:, :, :, test_indices])
        train_joints.append(joints_dataset[:, :, train_indices])
        test_joints.append(joints_dataset[:, :,test_indices])
        train_image.append(image_dataset[:, :, train_indices])
        test_image.append(image_dataset[:, :, test_indices])

    # Concatenate arrays along the last axis
    train_csi = np.concatenate(train_csi, axis=-1)
    test_csi = np.concatenate(test_csi, axis=-1)
    train_joints = np.concatenate(train_joints, axis=-1)
    test_joints = np.concatenate(test_joints, axis=-1)
    train_image = np.concatenate(train_image, axis=-1)
    test_image = np.concatenate(test_image, axis=-1)

    return train_csi, test_csi, train_joints, test_joints, train_image, test_image
