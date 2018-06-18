
"""
Manipulate the dataset for feature extraction [Pytorch]

@author:dang-khoa
"""

import torch
import numpy as np
import os.path as path
import skimage.io as io
import pandas as pd
import _pickle as pkl
from PIL import Image
from sklearn.model_selection import KFold

class AvitoImage(object):
    """
    Load Avito image.
    """
    def __init__(self, imlist, prob, root, transform=None):
        """
        Load dataset from a txt file
        Args:
            txtfile (string): list of image filename (without extension)
            root (string): the parent folder containing images
            transform: (torch.transform): preprocess image
        """

        self.txt = imlist
        self.prob = prob
        self.root = root
        self.transform = transform



    def __len__(self):
        return len(self.txt)


    def __getitem__(self, idx):
        # load image from
        WHITE_COLOR = (255, 255, 255)
        DEFAULT_SIZE = (200, 200)
        imid = self.txt[idx]
        prob = self.prob[idx]
        if type(imid) is float:
            # return a black image
            return {'image': self.transform(Image.new('RGB', DEFAULT_SIZE)),
                    'imid': str(imid), 'prob': prob}

        name, ext= path.splitext(imid)
        # all images are jpg files
        if ext == "":
            ext = ".jpg"

        impath = path.join(self.root, imid + ext )
        try:
            # print("Load {}".format(impath))
            im = Image.open(impath)
        except OSError as e:
            # return a white image
            # print("Error: {}".format(e))
            return {'image': self.transform(
                Image.new('RGB', DEFAULT_SIZE, WHITE_COLOR)),
                    'imid': imid, 'prob': prob}


        # apply transformation
        if self.transform:
            im = self.transform(im)

        sample = {'image': im, 'imid': imid, 'prob': prob}
        return sample

def preprocess_avito(csv_file, n_folds=5, fold_idx=0, is_train=True):
    """
    Load CSV file, pre-process data and generate the indexes
    Arguments:
        - csv_file (string): AVITO's csv file
        - is_train (bool): is the csv the training data
    """
    # load the data

    if is_train:
        use_cols = ['item_id', 'image', 'deal_probability']
        dtypes = {'deal_probability': 'float32', 'image': str}
    else:
        use_cols = ['item_id', 'image']
        dtypes = { 'image': str}


    data = pd.read_csv(csv_file, dtype=dtypes, index_col='item_id',
                            usecols=use_cols)

    data = data[0:200]
    if not is_train:
        return data.image.values

    folds = None
    # check if the indexes already saved
    indexes_path = path.join("checkpoint", "avito_indexes_"+str(n_folds)+".pkl")
    images = data.image.values
    probs = data.deal_probability.values
    if not path.exists(indexes_path):
        zeros = np.zeros((len(data), 1))
        kfold = KFold(n_splits=n_folds)
        folds = []
        for train_idx, val_idx in kfold.split(zeros):

            images_fold = {
                'train': images[train_idx],
                'val': images[val_idx]
            }

            probs_fold = {
                'train': probs[train_idx],
                'val': probs[val_idx]
            }

            folds.append({'images': images_fold, 'probs': probs_fold})

        with open(indexes_path, 'wb') as f:
            pkl.dump(folds, f)

    else:
        with open(indexes_path, 'rb') as f:
            folds = pkl.load(f)

    return folds[fold_idx]

