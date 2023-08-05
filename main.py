import os
import gc
import torch
import copy
import ResNet
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from dataloader import LeukemiaLoader
from dataloader import getData
from torch.utils.data import Dataset, DataLoader

def evaluate(model, loader_valid, device):
    model.eval()
    correct_img = 0
    for img, label in loader_valid:
        img = img.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        prediction = model(img)
        correct_img += prediction.max(dim=1)[1].eq(label).sum().item()
    
    avg_acc_test = correct_img / len(loader_valid.dataset)
    return avg_acc_test

def test(model_name, model, loader_test, device):
    print("test() not defined")

def train(model_name, loader_train, looader_valid, epochs, device, learning_rate):
    print(f'Training {model_name} ...')
    if model_name == 'ResNet18':
        model = ResNet.ResNet18(input_channels=3, num_classes=2)
    elif model_name == 'ResNet50':
        model = ResNet.ResNet50(input_channels=3, num_classes=2)
    else:
        model = ResNet.ResNet152(input_channels=3, num_classes=2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    Loss = nn.CrossEntropyLoss()
    
    df = pd.DataFrame()
    df['epoch'] = range(1, epochs+1)
    best_model_wts = None
    best_model_acc = 0
    acc_train = list()
    acc_test = list()
    for epoch in range(1, epochs+1):
        correct_img = 0
        model_loss = 0
        model.train()
        for img, label in loader_train:
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            prediction = model(img)
            # eq() return a bool tensor, if eq, then 1, else 0
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
            print(f'epoch : {epoch}, loss : {avg_model_loss}, accuracy : {acc_rate_train}%')
        """
            Validation
        """
        model.eval(model, epochs)
        acc_rate_eval = evaluate(model, loader_valid, device)
        acc_test.append(acc_rate_eval)
        if acc_rate_eval > best_model_acc:
            best_model_acc = acc_rate_eval
            best_model_wts = copy.deepcopy(model.state_dict())
        df[model_name+' train'] = acc_train
        df[model_name+' test'] = acc_test
        
        return df, best_model_wts
     
# Read img path(name) from ./csv_file, and use the name and predict value build a new csv file
def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./predict_result/110611008_resnet18.csv", index=False)
    
def show_result(df):
    fig = plt.figure(fig_size=(10, 6))
    for name in df.columns[1:]:
        plt.plot('epoch', name, data=df)
    plt.legend()
    return fig
###############################
    """
        HyperParameters
    """
###############################
batch_num = 256
epochs = 2
learning_rate = 0.01
###############################
"""
    main Function
"""
###############################
if __name__ == "__main__":
    torch.cuda.empty_cache()
    print("Good Luck :)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = input('Please input mode : ')
    if mode == 'train':
        dataset_train = LeukemiaLoader('train')
        loader_train = DataLoader(dataset_train, batch_num, shuffle=True, num_workers=0)
        dataset_valid = LeukemiaLoader('valid')
        loader_valid = DataLoader(dataset_train, batch_num, shuffle=True, num_workers=0)
        df_ResNet18, best_model_wts_ResNet18 = train('ResNet18', loader_train, loader_valid, epochs, device, learning_rate)
        df_ResNet50, best_model_wts_ResNet50 = train('ResNet50', loader_train, loader_valid, epochs, device, learning_rate)
        df_ResNet152, best_model_wts_ResNet152 = train('ResNet152', loader_train, loader_valid, epochs, device, learning_rate)
        # axis = 0 is concat on vertical direction, axis = 1 is concat on horizontal direction
        # if ignore_index = True, ignore origin index, reindexing
        df_all_ResNet = pd.concat([df_ResNet18, df_ResNet50, df_ResNet152], axis=1, ignore_index=True)
    else:
        print('Testing ...')
        dataset_test_restnet_18 = LeukemiaLoader('test_resnet_18')
        loader_test_restnet_18 = DataLoader(dataset_test_restnet_18, batch_num, shuffle=False, num_workers=0)
        test()
    figure = show_result(df_all_ResNet)
    figure.savefig('ResNet accuracy.png')
 