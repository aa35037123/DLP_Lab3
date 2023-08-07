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

def evaluate(model, loader_valid, device, num_classes):
    model.eval()
    correct_img = 0
    confusion_matrix = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for img, label in loader_valid:
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            result = model(img)
            prediction_class = result.max(dim=1)[1] # return a (1x1) tensor, represent indices of max value each row
            correct_img += prediction_class.eq(label).sum().item()
            for i in range(len(label)): # len(label) = batch_size
                confusion_matrix[int(label[i].item())][int(prediction_class[i].item())] += 1
        avg_acc_test = correct_img / len(loader_valid.dataset) * 100.
    # cuz each row will sum up to 1, so we need to sum up each row, and resape it into (num_classes, 1) 
    # then we can directly use it to divide confusion_matrix 
    # print(f'confusion before norm : {confusion_matrix}')
    confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=1).reshape(num_classes, 1) 
    return avg_acc_test, confusion_matrix

def test(model_name, loader_test, device, model_path):
    if model_name == 'resnet18':
        model = ResNet.ResNet18(input_channels=3, num_classes=2)
    elif model_name == 'resnet50':
        model = ResNet.ResNet50(input_channels=3, num_classes=2)
    else:
        model = ResNet.ResNet152(input_channels=3, num_classes=2)
    model.load_state_dict(torch.load(model_path))
        
    model.eval() # eval() can close Batch Normalization and Dropout
    model.to(device)
    predictions = list()
    with torch.no_grad():
        for img in loader_test:
            img = img.to(device, dtype=torch.float)
            result = model(img)
            prediction = result.max(dim=1)[1]
            for i in range(len(prediction)):
                predictions.append(prediction[i].item())
    
    return predictions

def train(model_name, loader_train, loader_valid, epochs, device, learning_rate, num_classes):
    print(f'Training {model_name} ...')
    if model_name == 'resnet18':
        model = ResNet.ResNet18(input_channels=3, num_classes=num_classes)
    elif model_name == 'resnet50':
        model = ResNet.ResNet50(input_channels=3, num_classes=num_classes)
    else:
        model = ResNet.ResNet152(input_channels=3, num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    Loss = nn.CrossEntropyLoss()
    
    df = pd.DataFrame()
    best_model_wts = None
    best_model_acc = 0.0
    acc_train = list()
    acc_test = list()
    for epoch in range(1, epochs+1):
        gc.collect()
        torch.cuda.empty_cache()
        correct_img = 0
        model_loss = 0.0
        model.train()
        # DataLoader will split all data into lots of batch
        # for each loop, there are number of batch size img and label 
        for img, label in loader_train:
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            prediction = model(img)
            # eq() return a bool tensor, if eq, then 1, else 0
            # max(dim=1):
            # The first tensor contains the maximum predicted values for each image.
            # The second tensor contains the indices of the maximum values, which correspond to the predicted class labels for each image.
            correct_img += prediction.max(dim=1)[1].eq(label).sum().item()
            # no matter what prediction shape is, if only batch size is the same, use loss fn can calculate gradient 
            # and use it on bp, improve every neuron in conv2d
            loss = Loss(prediction, label)
            model_loss += loss
            # Update network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_model_loss = model_loss / len(loader_train.dataset)
        acc_rate_train = correct_img / len(loader_train.dataset) * 100.
        acc_train.append(acc_rate_train)
        if epoch % 5 == 0:
            print(f'epoch : {epoch}, loss : {avg_model_loss}, accuracy : {acc_rate_train:.2f}%')
        """
            Validation
        """
        model.eval()
        acc_rate_eval, confunsion_matrix = evaluate(model, loader_valid, device, num_classes)
        acc_test.append(acc_rate_eval)
        if acc_rate_eval > best_model_acc:
            best_model_acc = acc_rate_eval
            best_model_wts = copy.deepcopy(model.state_dict())
    # print(f'Length of acc_train : {len(acc_train)}')
    # print(f'Length of acc_test : {len(acc_test)}')
    df[model_name+' train'] = acc_train
    df[model_name+' test'] = acc_test
    
    return df, best_model_wts, confunsion_matrix

# Read img path(name) from ./csv_file, and use the name and predict value build a new csv file
def save_result(csv_path, save_path, predict_result, model_name):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv(save_path, index=False)
    
def show_result(df):
    fig = plt.figure(figsize=(10, 6))
    for name in df.columns[1:]:
        plt.plot('epoch', name, data=df)
    plt.legend()
    return fig
def show_confusion(confusion_matrix):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.xaxis.set_label_position('top')
    # print(f'confusion matrix : {confusion_matrix}')
    for i in range(confusion_matrix.shape[0]): # i means row
        for j in range(confusion_matrix.shape[1]): # j means col
            # ax.text(x, y)
            ax.text(j, i, '{:.2f}'.format(confusion_matrix[i, j]), va='center', ha='center')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    return fig

def crop_test():
    img_path = './new_dataset/test/814.bmp'
    img = Image.open(img_path)
    img.show()
    crop_img = img.crop((0, 0, 300, 300))
    crop_img.show()
###############################
    """
        HyperParameters
    """
###############################
batch_num_ResNet18 = 64
batch_num_ResNet50 = 20
batch_num_ResNet152 = 8
epochs = 50
learning_rate = 0.005
num_classes = 2
root = os.getcwd()
epochs_lr = '50-0005'
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
###############################
"""
    main Function
"""
###############################
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    print("Good Luck :)")
    print('GPU is available now.') if torch.cuda.is_available() else print('GPU is not available now.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current device : {device}')
    mode = input('Please input mode : ')
    model_name = input('Please input model name you want to train/evaluate : ')

    if mode == 'train':
        dataset_train_ResNet = LeukemiaLoader('train')
        dataset_valid_ResNet = LeukemiaLoader('valid')
        ######
        # train ResNet18
        ######
        if model_name == 'resnet18':
            loader_train_ResNet18 = DataLoader(dataset_train_ResNet, batch_num_ResNet18, shuffle=True, num_workers=4)
            # print(f'Input img\'s shape : {dataset_train_ResNet18.__getitem__(13)[0].shape}')
            # print(f'loader_train\'s shape : {loader_train.shape}')
            loader_valid_ResNet18 = DataLoader(dataset_valid_ResNet, batch_num_ResNet18, shuffle=True, num_workers=4)
            df_ResNet18, best_wts_ResNet18, confusion_ResNet18 = train('resnet18', loader_train_ResNet18, loader_valid_ResNet18, 
                                                                        epochs, device, learning_rate, num_classes)
            print('Training of ResNet18 is finish !')
            df_ResNet18.to_csv(_path_df_ResNet18, index=False)
            torch.save(best_wts_ResNet18, _path_best_wts_ResNet18)
            fig_confusion_ResNet18 = show_confusion(confusion_ResNet18)
            fig_confusion_ResNet18.savefig(_path_confusion_ResNet18)
        ######
        # train ResNet50
        ######
        elif model_name == 'resnet50':
            loader_train_ResNet50 = DataLoader(dataset_train_ResNet, batch_num_ResNet50, shuffle=True, num_workers=4)
            loader_valid_ResNet50 = DataLoader(dataset_valid_ResNet, batch_num_ResNet18, shuffle=True, num_workers=4)
            df_ResNet50, best_wts_ResNet50, confusion_ResNet50 = train('resnet50', loader_train_ResNet50, loader_valid_ResNet50, 
                                                                        epochs, device, learning_rate, num_classes)
            print('Training of ResNet50 is finish !')
            df_ResNet50.to_csv(_path_df_ResNet50, index=False)
            torch.save(best_wts_ResNet50, _path_best_wts_ResNet50)
            fig_confusion_ResNet50 = show_confusion(confusion_ResNet50)
            fig_confusion_ResNet50.savefig(_path_confusion_ResNet50)
            # plt.show(fig_confusion_ResNet50)
        ######
        # train ResNet152
        ######
        else:
            loader_train_ResNet152 = DataLoader(dataset_train_ResNet, batch_num_ResNet152, shuffle=True, num_workers=4)
            loader_valid_ResNet152 = DataLoader(dataset_valid_ResNet, batch_num_ResNet152, shuffle=True, num_workers=4)
            df_ResNet152, best_wts_ResNet152, confusion_ResNet152 = train('resnet152', loader_train_ResNet152, loader_valid_ResNet152, 
                                                                        epochs, device, learning_rate, num_classes)
            print('Training of ResNet152 is finish !')
            df_ResNet152.to_csv(_path_df_ResNet152, index=False)
            torch.save(best_wts_ResNet152, _path_best_wts_ResNet152)
            fig_confusion_ResNet152 = show_confusion(confusion_ResNet152)
            fig_confusion_ResNet152.savefig(_path_confusion_ResNet152)

    else:
        print('Testing ...')
        ######
        # test ResNet18
        ######
        if model_name == 'resnet18':
            dataset_test_ResNet18 = LeukemiaLoader('test_resnet18')
            loader_test_ResNet18 = DataLoader(dataset_test_ResNet18, batch_num_ResNet18, shuffle=False, num_workers=4)
            predictions_ResNet18 = test(model_name, loader_test_ResNet18, device, _path_best_wts_ResNet18)
            save_result(csv_path=_path_test_file_ResNet18, save_path=_path_test_result_ResNet18, predict_result=predictions_ResNet18, model_name='resnet_18')
        ######
        # test ResNet50
        ######
        elif model_name == 'resnet50':
            dataset_test_ResNet50 = LeukemiaLoader('test_resnet50')
            loader_test_ResNet50 = DataLoader(dataset_test_ResNet50, batch_num_ResNet50, shuffle=False, num_workers=4)
            predictions_ResNet50 = test(model_name, loader_test_ResNet50, device, _path_best_wts_ResNet50)
            save_result(csv_path=_path_test_file_ResNet50, save_path=_path_test_result_ResNet50, predict_result=predictions_ResNet50, model_name='resnet_50')
        ######
        # test ResNet152
        ######
        else:
            dataset_test_ResNet152 = LeukemiaLoader('test_resnet152')
            loader_test_ResNet152 = DataLoader(dataset_test_ResNet152, batch_num_ResNet152, shuffle=False, num_workers=4)
            predictions_ResNet152 = test(model_name, loader_test_ResNet152, device, _path_best_wts_ResNet152)
            save_result(csv_path=_path_test_file_ResNet152, save_path=_path_test_result_ResNet152, predict_result=predictions_ResNet152, model_name='resnet_152')
    
    # axis = 0 is concat on vertical direction, axis = 1 is concat on horizontal direction
    # if ignore_index = True, ignore origin index, reindexing
    df_epochs = pd.DataFrame()
    df_epochs['epoch'] = range(1, epochs+1)
    # check if df_ResNet18 exist
    df_ResNet18 = pd.read_csv(_path_df_ResNet18) if os.path.exists(_path_df_ResNet18) else None
    df_ResNet50 = pd.read_csv(_path_df_ResNet50) if os.path.exists(_path_df_ResNet50) else None
    df_ResNet152 = pd.read_csv(_path_df_ResNet152) if os.path.exists(_path_df_ResNet152) else None
    
    df_all_ResNet = pd.concat([df_epochs, df_ResNet18, df_ResNet50, df_ResNet152], axis=1, ignore_index=False)
    figure = show_result(df_all_ResNet)
    figure.savefig(_path_acc_fig)
 