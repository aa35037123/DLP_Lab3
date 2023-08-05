import os
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data


def getData(mode):
    if mode == 'train':
        df = pd.read_csv('./csv_file/train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv('./csv_file/valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == 'test_resnet_18':
        df = pd.read_csv('./csv_file/resnet_18_test.csv')
        path = df['Path'].tolist()
        return path, None
    elif mode == 'test_resnet_50':
        df = pd.read_csv('./csv_file/resnet_50_test.csv')
        path = df['Path'].tolist()
        return path, None
    elif mode == 'test_resnet_152':
        df = pd.read_csv('./csv_file/resnet_152_test.csv')
        path = df['Path'].tolist()
        return path, None

class LeukemiaLoader(data.Dataset):
    def __init__(self, mode):
        """
        Args:
            mode : Indicate procedure status(training or valid or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.img_name, self.label = getData(mode)
        self.mode = mode
        # To tensor can Convert the pixel value to [0, 1]
        #               Transpose the image shape from [H, W, C] to [C, H, W]
        self.transformations = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip()])
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    # this function works different on test and train stage
    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        # load img
        # Image.open is used to get the picture and transform it into an obj
        img_obj = Image.open(self.img_name[index])
        # print(f'img : {img}')
        label = self.label[index]
        img = self.transformations(img_obj)
        return img, label