import os
import gc
import torch
import copy
import ResNet
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from dataloader import LeukemiaLoader
from dataloader import getData
from torch.utils.data import Dataset, DataLoader

def find_highest_acc(path_df, model_name):
    df = pd.read_csv(path_df) if os.path.exists(path_df) else None
    highest_acc = 0
    test_col_name = model_name +' test'
    for acc in df[test_col_name]:
        if acc > highest_acc:
            highest_acc = acc
    return highest_acc
root = os.getcwd()
epochs_lr = '30-001'
_path_df_ResNet18 = os.path.join(root, 'csv_file', 'df_ResNet18 '+epochs_lr+'.csv')
_path_df_ResNet50 = os.path.join(root, 'csv_file', 'df_ResNet50 '+epochs_lr+'.csv')
_path_df_ResNet152 = os.path.join(root, 'csv_file', 'df_ResNet152 '+epochs_lr+'.csv')
_path_acc_fig = os.path.join(root, 'figure', 'ResNet Accuracy '+epochs_lr+'.png')
_path_confusion_ResNet18 = os.path.join(root, 'figure', 'Confusion ResNet18 '+epochs_lr+'.png')
_path_confusion_ResNet50 = os.path.join(root, 'figure', 'Confusion ResNet50 '+epochs_lr+'.png')
_path_confusion_ResNet152 = os.path.join(root, 'figure', 'Confusion ResNet152 '+epochs_lr+'.png')
_path_best_wts_ResNet18 = os.path.join(root, 'model', 'ResNet18 '+epochs_lr+'.pt')
_path_best_wts_ResNet50 = os.path.join(root, 'model', 'ResNet50 '+epochs_lr+'.pt')
_path_best_wts_ResNet152 = os.path.join(root, 'model', 'ResNet152 '+epochs_lr+'.pt')
_path_test_file_ResNet18 = os.path.join(root, 'csv_file/resnet_18_test.csv')
_path_test_file_ResNet50 = os.path.join(root, 'csv_file/resnet_50_test.csv')
_path_test_file_ResNet152 = os.path.join(root, 'csv_file/resnet_152_test.csv')
_path_test_result_ResNet18 = os.path.join(root, 'csv_file/110611008_result_resnet18 '+epochs_lr+'.csv')
_path_test_result_ResNet50 = os.path.join(root, 'csv_file/110611008_result_resnet50 '+epochs_lr+'.csv')
_path_test_result_ResNet152 = os.path.join(root, 'csv_file/110611008_result_resnet152 '+epochs_lr+'.csv')
if __name__ == '__main__':
    model_name = input('Please input model name you want to evaluate : ')
    if model_name == 'resnet18':
        highest_acc = find_highest_acc(_path_df_ResNet18, model_name)
    elif model_name == 'resnet50':
        highest_acc = find_highest_acc(_path_df_ResNet50, model_name)
    else:
        highest_acc = find_highest_acc(_path_df_ResNet152, model_name)
    print(f'{model_name} test\'s best acc : {highest_acc:.2f}')