'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-24 13:22:11
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-28 08:43:26
FilePath: /pytorch_forward_forward/utils/misc.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchaudio

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    首先，函数创建了输入数据 x 的一个副本 x_,以确保在不改变原始数据的情况下进行修改。
    接着，函数将 x_ 的前10个像素位置(第0列至第9列)全部设置为0.0,意味着将前10个像素清零。
    然后，函数使用标签 y 来创建一个 one-hot 编码表示，将 y 对应的位置设为 x 的最大值。例如，如果标签 y 为 3,则会将 x_ 中第3个像素位置(索引为 3)设置为 x 的最大值。
    最后，函数返回修改后的数据 x_,其中标签信息已被叠加在前10个像素位置上,其余像素与输入数据 x 保持不变。
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_



class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        # 定义一个包含n个数据的列表
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def avg(self):
        return [sum(a)/len(a) for a in self.data]

    def __getitem__(self, idx):
        return self.data[idx]



def accuracy(y_hat, y):  # @save

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:

        y_hat = y_hat.argmax(axis=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class DataGenerator(Dataset):
 
    def __init__(self, path, kind='train'):
        if kind=='train':
            files = Path(path).glob('[1-3]-*')
        if kind=='val':
            files = Path(path).glob('4-*')
        if kind=='test':
            files = Path(path).glob('[4-5]-*')
        
        self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
        self.length = len(self.items)
        
    def __getitem__(self, index):
        filename, label = self.items[index]
        data_tensor, rate = torchaudio.load(filename)
        return (data_tensor, int(label))
    
    def __len__(self):
        return self.length
    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()