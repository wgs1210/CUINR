import argparse
import json
import random
from pathlib import Path
import os
from model import getModelDict
from datasets import dataset_dict
import numpy as np
import torch
import torchvision.transforms as transforms
import utils.misc as utils
# 加载float16_converter转换器
from onnxmltools.utils.float16_converter import convert_float_to_float16
# from torchsummary import summary
import zlib
from onnxruntime.quantization import QuantType, quantize_dynamic
import onnx
import onnxruntime

def float32ToUint8(model_fp32, model_quant_dynamic):

    from onnx import shape_inference,helper
    # 动态量化
    quantize_dynamic(
        model_input=model_fp32, # 输入模型
        model_output=model_quant_dynamic, # 输出模型
        weight_type=QuantType.QUInt8, # 参数类型 Int8 / UInt8
    )

def float32ToFloat16(output_onnx_name, f16onnname):
    onnx_model = onnx.load_model(output_onnx_name)
    onnx_model_fp16 = convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save_model(onnx_model_fp16, f16onnname)

def get_args_parse():

    parser = argparse.ArgumentParser('Dense NeRV', add_help=False)

    parser.add_argument('--cfg_path', default='', type=str, help='path to specific cfg yaml file path')
    parser.add_argument('--output_dir', default='', type=str, help='path to save the log and other files')
    parser.add_argument('--time_str', default='', type=str, help='just for tensorboard dir name')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--port', default=29500, type=int, help='port number')

    return parser

def main(args):
    args.distributed = False
    # get cfg yaml file
    cfg = utils.load_yaml_as_dict(args.cfg_path)
    # dump the cfg yaml file in output dir
    utils.dump_cfg_yaml(cfg, args.output_dir)

    #新添加的
    cfg['model']['unmixing']     = args.unmixing
    cfg['eval']         = args.eval
    print(cfg)

    device = torch.device(args.device)
    # device = 'cpu'

    # fix the seed
    seed = cfg['seed']
    # seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_dict = getModelDict(args)
    model = model_dict[cfg['model']['model_name']](cfg=cfg)
    params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    print(f'Model Params: {params}M')
    print(model)
    model.to(device)

    # pth_root = rf'{args.rootpath}\pathdir\xiongan'
    # pth_name = 'matiwan_vca_duibi'

    # pth_root = rf'{args.rootpath}\pathdir\pavia'
    # pth_name = 'pavia_vac'

    pth_root = rf'{args.rootpath}\pathdir\chikusei'
    pth_name = 'chikusei_vca'#soft

    # pth_root = rf'{args.rootpath}\pathdir\houston'
    # pth_name = 'houston_vca'

    pth_path = f'{pth_root}/{pth_name}.pth'
    savedparames = torch.load(pth_path, device)
    model.load_state_dict(savedparames)
    model.eval()

    img_transform = transforms.ToTensor()
    pic_path = args.fullpath

    model.eval()
    dataset_train = dataset_dict[cfg['dataset_type']](main_dir=pic_path, transform=img_transform, train=True, args=args)
    savedpath = rf"{args.matdir_dir}/{pth_name}.mat"
    dataset_train.getHsiWithOutOnnx(model, device, args, saved=True, savedpath=savedpath)

    #量化
    dummy_input = torch.rand(size=[1]).to(device)
    output_onnx_name = f"{pth_root}/onnx/{pth_name}.onnx"
    torch.onnx.export(model, dummy_input, output_onnx_name, export_params=True, opset_version=13, do_constant_folding=False)
    
    f16_onnx_name = f"{pth_root}/onnx/f16_{pth_name}.onnx"
    # float32ToUint8(output_onnx_name, f16_onnx_name)
    float32ToFloat16(output_onnx_name, f16_onnx_name)
    
    # 无损压缩，CUINR压缩的最后一步, 生成的.tmp即为最后的压缩文件
    deflate_out_path = f'{pth_root}/onnx/{pth_name}_deflate.tmp'
    deflate(f16_onnx_name, deflate_out_path)

    # CUINR开始解压
    deflate_decompress_path = f'{pth_root}/onnx/{pth_name}_deflate.onnx'
    inverseDeflate(deflate_out_path, deflate_decompress_path)

    # outonnx = f16_onnx_name
    outonnx = deflate_decompress_path

    # 创建一个ONNX Runtime的会话选项对象，用于指定设备
    providers = ['CUDAExecutionProvider'] if onnxruntime.get_device() == 'GPU' else ['CPUExecutionProvider']
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1  # 设置线程数，根据需要调整
 
    ort_session = onnxruntime.InferenceSession(outonnx, providers=providers, sess_options=session_options)  
    input_name = ort_session.get_inputs()[0].name  
    print('Input Name:', input_name)    

    savedpath = rf"{args.matdir_dir}/f16_{pth_name}.mat"
    dataset_train.getHsiWithOnnxSession(ort_session, input_name, device, args, saved=False, savedpath=savedpath)

def deflate(inputfile, outputpath):
    with open(inputfile, 'rb') as file:
        content = file.read()

    compressed_content = zlib.compress(content)
    with open(outputpath, 'wb') as compressed_file:
        compressed_file.write(compressed_content)

def inverseDeflate(input_path, output_path):

    with open(input_path, 'rb') as file:
        content = file.read()
    
    decompress_data = zlib.decompress(content)

    with open(output_path, 'wb') as decompressed_file:
        decompressed_file.write(decompress_data)

def defineConfig(args):

    args.ours       = True

    if args.ours:
        args.unmixing   = True
    else:
        args.unmixing   = False

rootpath = r'C:\Users\jj\Documents\project\python_server\CUINR'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('E-NeRV training and evaluation script', parents=[get_args_parse()])
    args = parser.parse_args()
    args.rank = 0
    args.ranks = [0, 1, None]
    args.device = f'cuda:{args.rank}'
    args.iomat = True
    args.eval = True

    # args.dataname = 'xiongan'
    # data_path = f'{rootpath}/data/matiwan'
    # train_liebiao = ['matiwan_vca']
    # args.cfg_path = f'{rootpath}/cfgs/github/cuinr_matiwan.yaml'

    # args.dataname = 'pavia'
    # data_path = f'{rootpath}/data/pavia'
    # train_liebiao = ['pavia_vca']
    # args.cfg_path = f'{rootpath}/cfgs/github/cuinr_pavia.yaml'

    args.dataname = 'chikusei'
    data_path = f'{rootpath}/data/chikusei'
    train_liebiao = ['chikusei_vca']
    args.cfg_path = f'{rootpath}/cfgs/github/cuinr_chikusei_4.yaml'

    # args.dataname = 'houston'
    # data_path = f'{rootpath}/data/houston'
    # train_liebiao = ['houston_vca']
    # args.cfg_path = f'{rootpath}/cfgs/github/cuinr_houston.yaml'
    
    defineConfig(args)

    assert args.cfg_path is not None, 'Need a specific cfg path!'
    args.output_dir = f'{rootpath}/pathdir/{args.dataname}'
    args.matdir_dir = f'{rootpath}/testout/{args.dataname}'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.matdir_dir:
        Path(args.matdir_dir).mkdir(parents=True, exist_ok=True)

    args.filename = ''
    args.fullpath = ''
    args.rootpath = rootpath
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            mat_path = f'{root}/{file}'
            mat_name = file.split('.')[0]
            if mat_name not in train_liebiao:
                continue
            args.fullpath = mat_path
            args.filename = mat_name
            main(args)