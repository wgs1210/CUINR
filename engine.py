import math
import os
import sys
import torch
import utils.misc as utils
import torch.nn.functional as F
from datetime import datetime

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    cfg,
    args,
    datasize,
    start_time,
    writer=None,
    dataset = None,
    sum2one = None,
    scheduler = None,
):
    model.train()
    epoch_start_time = datetime.now()
    loss_type = cfg["loss"]

    psnr_list = []
    msssim_list = []

    isJump = cfg['isjump']
    # print('dataloader ', len(dataloader))
    train_loss = 0
    for i, data in enumerate(dataloader):
        
        #vca初始化
        channel_index = data['number'].numpy()[0]
        if isJump:
            # current_idx = data['number'].numpy()[0]
            if channel_index not in dataset.getTrainList():
                continue

        if args.ours:
            if args.isvca[channel_index] == 0:
                # print('no vca')
                args.isvca[channel_index] = 1
                model_dict = model.state_dict()
                vcaitem = data['vca_item'].squeeze(0).unsqueeze(2).unsqueeze(3)
                model_dict['unmixing.decoder.0.weight'] = vcaitem
                model.load_state_dict(model_dict)

        data = utils.to_cuda(data, device)
        # forward pass
        output_list, abu, = model(data)  # output is a list for the case that has multiscale
        additional_loss_item = {}

        if isinstance(output_list, dict):
            for k, v in output_list.items():
                if "loss" in k:
                    additional_loss_item[k] = v
            output_list = output_list["output_list"]
            abu = output_list["abu"]

        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list
        ]

        # target_list = [data["img_gt"]]

        loss_list = utils.loss_compute(output_list, target_list, loss_type)

        if len(abu) > 0 and cfg['model']['useSoftMax'] is False:
            loss_unmixing = [sum2one(abu[0])*float(cfg['asc_delta'])]# chikusei houston
            # loss_unmixing = [sum2one(abu[0])]#matiwan pavia
            loss_all = loss_unmixing + loss_list
            losses = sum(loss_all)
        else:
            losses = sum(loss_list)

        if len(additional_loss_item.values()) > 0:
            losses = losses + sum(additional_loss_item.values())

        if scheduler is None:
            lr = utils.adjust_lr(optimizer, epoch, cfg["epoch"], i, datasize, cfg)
        else:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))
        msssim_list.append(utils.msssim_fn(output_list, target_list))

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:
            train_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            train_psnr = torch.mean(train_psnr, dim=0)  # (num_stage)
            train_msssim = torch.cat(msssim_list, dim=0)  # (batchsize, num_stage)
            train_msssim = torch.mean(train_msssim.float(), dim=0)  # (num_stage)
            time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            # if not hasattr(args, "rank"):
            #     print_str = "[{}] Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
            #         time_now_string,
            #         epoch + 1,
            #         cfg["epoch"],
            #         i + 1,
            #         len(dataloader),
            #         lr,
            #         utils.RoundTensor(train_psnr, 2, False),
            #         utils.RoundTensor(train_msssim, 4, False),
            #     )
            #     for k, v in additional_loss_item.items():
            #         print_str += f", {k}: {v.item():.6g}"
                # print(print_str, flush=True)

            # elif args.rank in [0, None]:
            # elif args.rank in args.ranks:
            #     print_str = "[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
            #         time_now_string,
            #         args.rank,
            #         epoch + 1,
            #         cfg["epoch"],
            #         i + 1,
            #         len(dataloader),
            #         lr,
            #         utils.RoundTensor(train_psnr, 2, False),
            #         utils.RoundTensor(train_msssim, 4, False),
            #     )
                # print(print_str, flush=True)

    #调整lr
    if scheduler is not None:
        if epoch > 0 and epoch % 10 == 0:
            print('run scheduler step... ', epoch)
            scheduler.step()
            print('run scheduler....', optimizer.state_dict()['param_groups'][0]['lr'])
    mean_loss = train_loss / len(dataloader)

    train_stats = {
        "train_psnr": train_psnr,
        "train_msssim": train_msssim,
        "mean_loss":mean_loss,
    }
    if hasattr(args, "distributed") and args.distributed:
        train_stats = utils.reduce_dict(train_stats)

    # ADD train_PSNR TO TENSORBOARD
    if not hasattr(args, "rank"):
        h, w = output_list[-1].shape[-2:]
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
        for k, v in additional_loss_item.items():
            writer.add_scalar(f"Train/{k}", v.item(), epoch + 1)
        for (k, m) in model.named_modules():
            if isinstance(m, torch.nn.Module) and hasattr(m, "Lip_c"):
                writer.add_scalar(f"Stat/{k}_c", m.Lip_c[0].item(), epoch + 1)
                writer.add_scalar(f"Stat/{k}_w", m.abssum_max, epoch + 1)

    elif args.rank in [0, None] and writer is not None:
        h, w = output_list[-1].shape[-2:]
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
    epoch_end_time = datetime.now()
    print(
        "Epoch/Time: Epoch:{} \tCurrent:{:.2f} \tAverage:{:.2f}".format(
            epoch,
            (epoch_end_time - epoch_start_time).total_seconds(),
            (epoch_end_time - start_time).total_seconds() / (epoch + 1),
        )
    )

    return train_stats

@torch.no_grad()
def self_savetmp(model, dataset, device, args, saved=False, saveName='default'):
    model.eval()
    return dataset.getHSI(model, device, args, saved, saveName)
    # model.train()

@torch.no_grad()
def evaluate(model, dataloader, device, cfg, args, save_image=False):
    val_start_time = datetime.now()
    model.eval()

    psnr_list = []
    msssim_list = []

    gts = []
    covers = []

    for i, data in enumerate(dataloader):
        
        data = utils.to_cuda(data, device)
        # forward pass
        output_list, _ = model(data)  # output is a list for the case that has multiscale
        if isinstance(output_list, dict):
            output_list = output_list["output_list"]  # ignore the loss in eval
        torch.cuda.synchronize()

        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list
        ]

        # target_list = [data["img_gt"]]

        gts.append(data["img_gt"])
        covers.append(output_list[0])

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))
        msssim_list.append(utils.msssim_fn(output_list, target_list))

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:
            val_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            val_psnr = torch.mean(val_psnr, dim=0)  # (num_stage)
            val_msssim = torch.cat(msssim_list, dim=0)  # (batchsize, num_stage)
            val_msssim = torch.mean(val_msssim.float(), dim=0)  # (num_stage)
            time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            if not hasattr(args, "rank"):
                print_str = "[{}], Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                # print(print_str, flush=True)

            # elif args.rank in [0, None]:
            elif args.rank in args.ranks:
                print_str = "[{}] Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    args.rank,
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                # print(print_str, flush=True)

    gtHSI = torch.cat(gts, 1)
    coverHSI = torch.cat(covers, 1)
    psnrall = utils.psnr_fn([gtHSI], [coverHSI])
    # print('eval all ', psnrall)

    val_stats = {
        "val_psnr": val_psnr,
        "val_msssim": val_msssim,
        'val_psnrall':psnrall,
    }
    if hasattr(args, "distributed") and args.distributed:
        val_stats = utils.reduce_dict(val_stats)
    val_end_time = datetime.now()
    print(
        "Time on evaluate: \t{:.2f}".format(
            (val_end_time - val_start_time).total_seconds()
        )
    )

    return val_stats
