import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as io
import utils.misc as utils
import h5py
import scipy.io as scio
from basedata import basedata

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, train=True, args=None):
        self.main_dir = main_dir
        self.transform = transform
        frame_idx, self.frame_path = [], []

        self.datatool = basedata(args)

        self.all_imgs, self.all_imgs_idx, self.vca_items = self.create_cave_frame(main_dir)
        
        num_frame = 0 
        for img_id in self.all_imgs:
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)  # if 135 frames in total, this list will store 0, 1, 2, ..., 133, 134
            num_frame += 1          

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        # the id for first frame is 0 and the id for last is 1
        self.frame_idx = []
        for i in range(len(frame_idx)):
            x = frame_idx[i]
            self.frame_idx.append(float(x) / (len(frame_idx) - 1))

        # self.height = 720
        # self.width = 1280
        #新增
        self.init_(len(self.all_imgs))

    def __len__(self):
        return len(self.frame_idx)
    
    def __getitem__(self, idx):
        return self.train_data[idx]
    
    def createTrainData(self):
        trainData = []
        for idx, frame0_1 in enumerate(self.frame_idx):
            tensor_image = self.all_imgs[idx]
            
            frame_idx = torch.tensor(self.frame_idx[idx])
            vca_item = torch.tensor(self.vca_items[idx])

            data_dict = {
                "img_id": frame_idx,
                "img_gt": tensor_image,
                "number": idx,#新增
                "vca_item":vca_item,#新增
            }
            trainData.append(data_dict)

        return trainData
    
    def create_cave_frame(self,img_path):
        cave_frames = []
        cave_frames_index = []
        vca_items = []

        matFile, vac = self.datatool.loadMatFile(img_path)

        self.matFile = matFile#31,512,512
        c, h, w = matFile.shape

        for channel in range(c):
            cave_frames_index.append(channel)

            c_item = torch.tensor(matFile[channel:channel+1,:,:], dtype=torch.float32)
            cave_frames.append(c_item)

            vca_item =  torch.tensor(vac[channel:channel+1,:], dtype=torch.float32)
            vca_items.append(vca_item)
            
        return cave_frames, cave_frames_index, vca_items
    
    def init_(self, channel):
        self.train_list = self.datatool.defineTrainData(channel)
        self.frame_dic = self.define_frames()
        self.train_data = self.createTrainData()

    def getTrainList(self):
        return self.train_list
    
    def getFrames(self):
        return self.frame_dic

    def getHsiWithOnnxSession(self, session, input_name, device, args, saved=False, savedpath='default'):
        hsi = []
        for index, data in enumerate(self.train_data):
            tmpData = {}
            tmpData['number'] = torch.tensor([data['number']])
            tmpData['img_id'] = data['img_id'].unsqueeze(0) 
            tmpData['img_gt'] = data['img_gt']
            cudaData = tmpData['img_id'].numpy()#.astype(np.float16)
            output = session.run(None, {input_name:cudaData})
            hsi.append(torch.from_numpy(output[0]))
        
        recHSI = torch.cat(hsi, 1)
        gtHSI = self.datatool.loadGt(device, self.main_dir, args).cpu()

        psnr = utils.psnr_fn([recHSI], [gtHSI])
        print('ONNX重建的psnr_net ', psnr, 'shape ', recHSI.shape)
        utils.get_ssim(recHSI, gtHSI)

        if saved:
            coverHSI = recHSI.squeeze(0).detach().cpu().numpy()
            scio.savemat(savedpath, {'truth': coverHSI})

    def getHsiWithOutOnnx(self, model, device, args, saved=False, savedpath='default'):
        # self.startONNX(device, model, 0)
        hsi = []
        for index, data in enumerate(self.train_data):
            cudaData = data['img_id'].unsqueeze(0).to(device)#.to(device) 
            output, _ = model( cudaData )
            hsi.append(output[0].detach().cpu())
        
        recHSI = torch.cat(hsi, 1)
        gtHSI = self.datatool.loadGt(device, self.main_dir, args).detach().cpu()

        psnr = utils.psnr_fn([recHSI], [gtHSI])
        print('重建的psnr_net ', psnr, 'shape ', recHSI.shape)
        utils.get_ssim(recHSI, gtHSI)
        if saved:
            coverHSI = recHSI.squeeze(0).detach().cpu().numpy()
            scio.savemat(savedpath, {'truth': coverHSI})

    def get_frame_idx(self):
        return self.frame_idx

    def define_frames(self):
        frame_dic = {}
        for index, frame in enumerate(self.frame_idx):
            frame_dic[index] = frame
        return frame_dic
    
