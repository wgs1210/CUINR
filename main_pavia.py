import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import os
from model import getModelDict
from datasets import dataset_dict
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, dataloader
from torch.utils.data.distributed import DistributedSampler
from engine import train_one_epoch, evaluate, self_savetmp
from torch.utils.tensorboard import SummaryWriter
import utils.misc as utils
from model.E_NeRV_pavia import SumToOneLoss
import json

def get_args_parse():
    parser = argparse.ArgumentParser('Dense NeRV', add_help=False)

    parser.add_argument('--cfg_path', default='', type=str, help='path to specific cfg yaml file path')
    parser.add_argument('--output_dir', default='', type=str, help='path to save the log and other files')
    parser.add_argument('--time_str', default='', type=str, help='just for tensorboard dir name')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--port', default=29500, type=int, help='port number')

    return parser

def main(args):
    utils.init_distributed_mode(args)
    print('git:\n {}\n'.format(utils.get_sha()))

    # get cfg yaml file
    cfg = utils.load_yaml_as_dict(args.cfg_path)
    # dump the cfg yaml file in output dir
    utils.dump_cfg_yaml(cfg, args.output_dir)

    #新添加的
    cfg['model']['unmixing']    = args.unmixing
    cfg['eval']                 = args.eval

    if args.addepoch:
        cfg['optim']['lr_point']    = args.warm
        cfg['epoch']                = args.epoch
    
    print(cfg)

    device = torch.device(args.device)

    # fix the seed
    seed = cfg['seed']
    # seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_dict = getModelDict(args)
    model = model_dict[cfg['model']['model_name']](cfg=cfg)
    model.to(device)

    model_without_ddp = model
    
    if args.rank in args.ranks:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
        print(f'{args}\n {model}\n Model Params: {params}M')
        # tensorboard writer
        writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard_{}'.format(args.time_str)))
    
    print('是否跨通道训练 ', cfg['isjump'])
    img_transform = transforms.ToTensor()
    pic_path = args.fullpath

    dataset_train = dataset_dict[cfg['dataset_type']](main_dir=pic_path, transform=img_transform, train=True, args=args)
    dataset_val = dataset_dict[cfg['dataset_type']](main_dir=pic_path, transform=img_transform, train=False, args=args)
    # follow nerv implementation on sampler and dataloader
    sampler_train = DistributedSampler(dataset_train) if args.distributed else None
    sampler_val = DistributedSampler(dataset_val) if args.distributed else None
    sum2one = SumToOneLoss().to(device)
    dataloader_train = DataLoader(
        dataset_train, batch_size=cfg['train_batchsize'], shuffle=(sampler_train is None), num_workers=cfg['workers'], 
        pin_memory=True, sampler=sampler_train, drop_last=True, worker_init_fn=utils.worker_init_fn
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=cfg['val_batchsize'], shuffle=False, num_workers=cfg['workers'], 
        pin_memory=True, sampler=sampler_val, drop_last=False, worker_init_fn=utils.worker_init_fn
    )

    datasize = len(dataset_train)
    
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad],
            "lr": cfg['optim']['lr'],
        }
    ]

    optim_cfg = cfg['optim']
    if optim_cfg['optim_type'] == 'Adam':
        optimizer = optim.Adam(param_dicts, lr=optim_cfg['lr'], betas=(optim_cfg['beta1'], optim_cfg['beta2']))
    else:
        optimizer = None
    assert optimizer is not None, "No implementation of Optimizer!"
    
    scheduler = None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    output_dir = Path(args.output_dir)

    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]

    print('Start training')
    start_time = datetime.now()
    train_psnr = []
    for epoch in range(cfg['epoch']):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(#dataset_train新增
            model, dataloader_train, optimizer, device, epoch, cfg, args, datasize, start_time, writer, dataset_train, sum2one,scheduler
        )

        train_best_psnr = train_stats['train_psnr'][-1] if train_stats['train_psnr'][-1] > train_best_psnr else train_best_psnr
        train_best_msssim = train_stats['train_msssim'][-1] if train_stats['train_msssim'][-1] > train_best_msssim else train_best_msssim
        
        train_psnr.append(round(train_stats['train_psnr'][-1].item(), 2))
        # print('train psnr')
        # print(train_psnr)

        # if args.rank in [0, None]:
        if args.rank in args.ranks:
            print_str = '\ttraining: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(train_stats['train_psnr'][-1].item(), 
            train_best_psnr.item(), train_best_msssim.item())
            print(print_str, flush=True)
        
        checkpoint_paths = [output_dir / f'checkpoint_{args.filename}_{args.update}.pth']  # save one per epoch /代表拼接的意思
        if args.saveModel:
            # torch.save(model_without_ddp, f'{rootpath}/pathdir/{args.dataname}/total.pth')
            for checkpoint_path in checkpoint_paths:
                #模型存储与读取
                # state_dict=torch.load('net_params.pth')
                # model.load_state_dict(state_dict)
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': cfg,
                    'train_best_psnr': train_best_psnr,
                    'train_best_msssim': train_best_msssim,
                    'val_best_psnr': val_best_psnr,
                    'val_best_msssim': val_best_msssim,
                }, checkpoint_path)
        
        # evaluation
        '''
        if (epoch + 1) % cfg['eval_freq'] == 0 or epoch > cfg['epoch'] - 10:

            val_stats = evaluate(
                model, dataloader_val, device, cfg, args, save_image=False  # TODO: implement the save image
            )
            
            val_best_psnr = val_stats['val_psnr'][-1] if val_stats['val_psnr'][-1] > val_best_psnr else val_best_psnr
            val_best_msssim = val_stats['val_msssim'][-1] if val_stats['val_msssim'][-1] > val_best_msssim else val_best_msssim
            # if args.rank in [0, None]:
            if args.rank in args.ranks:
                print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                print_str += '\tevaluation: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}'.format(
                    val_stats['val_psnr'][-1].item(), val_best_psnr.item(), val_best_msssim.item())
                print(print_str)
                
                psnrs.append(round(val_stats['val_psnr'][-1].item(), 2))
                psnrall.append(round(val_stats['val_psnrall'][-1].item(), 2))
        '''
    torch.save(model.state_dict(), f'{rootpath}/pathdir/{args.dataname}/{args.update}.pth')
    print("Training complete in: " + str(datetime.now() - start_time))

def defineConfig(args):

    args.addepoch   = False
    args.addpara    = False
    args.eval       = False

    args.ours       = True

    if args.ours:
        args.unmixing   = True
    else:
        args.unmixing   = False

    args.update = f'{args.dataname}_ours_matiwan_test_{args.rank}'

def init_vca_flag(args, bandnumber):
    args.isvca = {}
    for c in range(bandnumber):
        args.isvca[c] = 0

rootpath = r'C:\Users\jj\Documents\project\python_server\CUINR_main'
if __name__ == '__main__':
    parser = argparse.ArgumentParser('E-NeRV training and evaluation script', parents=[get_args_parse()])
    args = parser.parse_args()
    args.rank = 0
    args.ranks = [0, 1, None]
    args.device = f'cuda:{args.rank}'
    args.iomat = True

    # args.dataname = 'xiongan'
    # args.bandnumber = 256
    # data_path = f'{rootpath}/data/matiwan'
    # train_liebiao = ['matiwan_vca']
    # args.iomat = True
    # args.cfg_path = f'{rootpath}/cfgs/github/cuinr_matiwan.yaml'

    # args.dataname = 'pavia'
    # args.bandnumber = 102
    # data_path = f'{rootpath}/data/pavia'
    # train_liebiao = ['pavia_vca']
    # args.cfg_path = f'{rootpath}/cfgs/github/cuinr_pavia.yaml'

    args.dataname = 'chikusei'
    args.bandnumber = 128
    data_path = f'{rootpath}/data/chikusei'
    train_liebiao = ['chikusei_vca']
    args.cfg_path = f'{rootpath}/cfgs/github/cuinr_chikusei_4.yaml'

    # args.dataname = 'houston'
    # args.bandnumber = 50
    # data_path = f'{rootpath}/data/houston'
    # train_liebiao = ['choustonuint16_vca']
    # args.cfg_path = f'{rootpath}/cfgs/github/cuinr_houston.yaml'

    defineConfig(args)

    assert args.cfg_path is not None, 'Need a specific cfg path!'
    args.output_dir = f'{rootpath}/pathdir/{args.dataname}'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.filename = ''
    args.fullpath = ''
    args.saveModel = False
    args.rootpath = rootpath
    init_vca_flag(args, args.bandnumber)

    for root, dirs, files in os.walk(data_path):
        for file in files:
            mat_path = f'{root}/{file}'
            mat_name = file.split('.')[0]
            if mat_name not in train_liebiao:
                continue
            args.fullpath = mat_path
            args.filename = mat_name
            main(args)
