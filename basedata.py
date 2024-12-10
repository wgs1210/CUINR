import numpy as np
import scipy.io as io
import h5py
import torch 

class basedata():

    def __init__(self, args) -> None:
        self.dataname = args.dataname
        self.args = args
       
    def defineTrainData(self, channel):
        train_list = self.define_train_pavia(channel)
        return train_list
    
    def loadMatFile(self, img_path):
        if self.args.iomat:
            data = io.loadmat(img_path)
            matFile = data['truth']#102,512,512
            hvca = data['HVca']#102,512,512
        else:
            data = h5py.File(img_path)['truth']#31,512,512
            matFile = data['truth'][:]
            hvca = data['HVca'][:]
        
        _max = np.max(matFile)
        matFile = matFile / _max 
        #matiwan, houston
        matFile = np.transpose(matFile, (2,0,1))
        #chikusei pavia
        return matFile, hvca

    def define_train_pavia(self, channel):
        train_list = []
        every = 8
        train_list = np.arange(0, channel, every)
        # print('训练的通道都有：', train_list)
        # print(f'{self.dataname}训练几个通道: ', len(train_list), ' every ', every)
        return train_list
    
    def loadGt(self, device, datapath, args):#datapath=self.main_dir 
        matFile, _ = self.loadMatFile(datapath)#256,512,512
        gtHSI = torch.tensor(matFile, dtype=torch.float32).to(device).unsqueeze(0)
        return gtHSI