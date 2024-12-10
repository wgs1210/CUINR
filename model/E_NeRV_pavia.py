##------------------------------------------------------------##
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from .model_utils import ActivationLayer, NormLayer, PositionalEncoding, gradient
from .NeRV_pavia import NeRV_MLP, NeRVBlock, Conv_Up_Block
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()#dim_head=64
        # dim_head = 12
        inner_dim = heads * dim_head
        project_out = not(heads==1 and dim_head==dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., prenorm=False):
        super(TransformerBlock, self).__init__()
        if prenorm:
            self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        else:
            self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x


class E_NeRV_Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.isEval = cfg['eval']
        data_name   = cfg['data_name']

        cfg = cfg['model']

        self.addGuide      = cfg['add_guide']            
        self.addUnmixing   = cfg['unmixing'] 

        if self.addUnmixing:
            # unmixing reconstruction module
            self.unmixing = AutoEncoder(1, cfg['lower_width'], cfg['useSoftMax'], data_name, self.isEval)

        # t mapping
        self.pe_t = PositionalEncoding(
            pe_embed_b=cfg['pos_b'], pe_embed_l=cfg['pos_l']
        )

        #stem_dim_num=512 #fc_hw_dim=16,16,196 fc_h=16 fc_w=16 fc_dim=196
        stem_dim_list = [int(x) for x in cfg['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in cfg['fc_hw_dim'].split('_')]
        self.block_dim = cfg['block_dim']#256->128
        #160,512,256
        mlp_dim_list = [self.pe_t.embed_length] + stem_dim_list + [self.block_dim]
        self.stem_t = NeRV_MLP(dim_list=mlp_dim_list, act=cfg['act'])

        # xy mapping
        xy_coord = torch.stack( 
            torch.meshgrid(
                torch.arange(self.fc_h) / self.fc_h, torch.arange(self.fc_w) / self.fc_w
            ), dim=0
        ).flatten(1, 2)  # [2, h*w]
        self.xy_coord = nn.Parameter(xy_coord, requires_grad=False)
        self.pe_xy = PositionalEncoding(
            pe_embed_b=cfg['xypos_b'], pe_embed_l=cfg['xypos_l']
        )
        #block_dim=256 #mlp_dim=128
        self.stem_xy = NeRV_MLP(dim_list=[2 * self.pe_xy.embed_length, self.block_dim], act=cfg['act'])

        self.trans1 = TransformerBlock(
            dim=self.block_dim, heads=1, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )
        self.trans2 = TransformerBlock(
            dim=self.block_dim, heads=8, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )

        if self.block_dim == self.fc_dim:
            self.toconv = nn.Identity()
        else:
            self.toconv = NeRV_MLP(dim_list=[self.block_dim, self.fc_dim], act=cfg['act'])
        
        # BUILD CONV LAYERS
        # self.layers, self.head_layers, self.norm_layers = [nn.ModuleList() for _ in range(3)]

        self.layers, self.head_layers, self.t_layers, self.t_layers_2, self.norm_layers = [nn.ModuleList() for _ in range(5)]
        # self.layers, self.head_layers, self.t_layers_2, self.norm_layers = [nn.ModuleList() for _ in range(4)]
        ngf = self.fc_dim #ngf=196
        ori128 = 128#128
        # ori128 = 128#128
        for i, stride in enumerate(cfg['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * cfg['expansion'])#new_ngf=588
            else:
                # change the channel width for each stage #lower_width=96
                new_ngf = max(ngf // (1 if stride == 1 else cfg['reduction']), cfg['lower_width'])
            
            if self.addGuide:
                self.t_layers_2.append(NeRV_MLP(dim_list=[ori128, 2*ngf], act=cfg['act']))#nerv mlp
            else:
                self.t_layers_2.append(nn.Identity())

            self.t_layers.append(nn.Identity())

            self.norm_layers.append(nn.InstanceNorm2d(ngf, affine=False))
            
            if i == 0:#
                self.layers.append(Conv_Up_Block(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            else:
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            
            #第一个block不加中间通道
            # self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            
            ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if cfg['sin_res']:
                if i == len(cfg['stride_list']) - 1:
                    if self.addUnmixing:
                        head_layer = nn.Identity()
                    else:
                        head_layer = nn.Conv2d(ngf, 1, 1, 1, bias=cfg['bias'])
                else:
                    head_layer = None
            else:
                if self.addUnmixing:
                    head_layer = nn.Identity()
                else:
                    head_layer = nn.Conv2d(ngf, 1, 1, 1, bias=cfg['bias'])

            self.head_layers.append(head_layer)
        self.sigmoid = cfg['sigmoid']

        self.T_num = 20
        
        if self.addGuide:
            self.pe_t_manipulate = PositionalEncoding(pe_embed_b=cfg['pos_b_tm'], pe_embed_l=cfg['pos_l_tm'])
            # self.t_branch = NeRV_MLP(dim_list=[self.pe_t_manipulate.embed_length, 128, 128], act=cfg['act'])
            self.t_branch = NeRV_MLP(dim_list=[self.pe_t_manipulate.embed_length, ori128, ori128], act=cfg['act'])

        self.loss = cfg['additional_loss'] if cfg.__contains__('additional_loss') else None
        self.loss_w = cfg['additional_loss_weight'] if cfg.__contains__('additional_loss_weight') else 1.0
        self.mse = nn.MSELoss()
    
    def fuse_t(self, x, t):#x=1, 196, 8, 8 t=1,392
        # x: [B, C, H, W], normalized among C
        # t: [B, 2* C]
        f_dim = t.shape[-1] // 2#f_dim=196
        gamma = t[:, :f_dim]#gamma=1,196
        beta = t[:, f_dim:]#beta=1,196

        gamma = gamma[..., None, None]#1,196,1,1
        beta = beta[..., None, None]#1,196,1,1
        out = x * gamma + beta
        return out

    def forward_impl(self, input_id):
        t = input_id

        a = self.pe_t(t)#1,160
        t_emb = self.stem_t(a) # [B, L] t_emb=1,256

        if self.addGuide:
            #时间引导pe_t_manipulate时间嵌入t_branch mlp==>Ct
            xx = self.pe_t_manipulate(t)
            t_manipulate = self.t_branch(xx)

        xy_coord = self.xy_coord#2,64
        x_coord = self.pe_xy(xy_coord[0])    # [h*w, C] 64,160
        y_coord = self.pe_xy(xy_coord[1])    # [h*w, C] 64,160
        xy_emb = torch.cat([x_coord, y_coord], dim=1)#64,320
        xy_emb = self.stem_xy(xy_emb).unsqueeze(0).expand(t_emb.shape[0], -1, -1)  # [B, h*w, L]
        xy_emb = self.trans1(xy_emb)#空间上下文
        # fuse t into xy map
        t_emb_list = [t_emb for i in range(xy_emb.shape[1])]#cuinr_d=t_emb.shape
        # print('cuinr d: ', t_emb.shape[1])
        t_emb_map = torch.stack(t_emb_list, dim=1)  # [B, h*w, L]

        emb = xy_emb * t_emb_map#t和坐标相乘
        emb = self.toconv(self.trans2(emb))#Ftheta融合时空#cuinr_C0=emb.shape
        # print('cuinr C0: ', emb.shape[2])
        emb = emb.reshape(emb.shape[0], self.fc_h, self.fc_w, emb.shape[-1])
        emb = emb.permute(0, 3, 1, 2)
        output = emb

        index = 0
        out_list = []
        abu_list = []

        for layer, head_layer, t_layer, t_layer_2, norm_layer in zip(self.layers, self.head_layers, self.t_layers, self.t_layers_2, self.norm_layers):
 
            output = norm_layer(output)

            if self.addGuide == True:
                t_feat = t_layer_2(t_manipulate)##cuinr_d0=t_feat.shape
                # print('cuinr d0: ', t_feat.shape[1])
                output = self.fuse_t(output, t_feat)

            output = layer(output)

            if self.addUnmixing:
            # 加入解混
                if head_layer is not None:
                    # print('cuinr C1: ', output.shape[1])
                    abu, rec = self.unmixing(output)
                    img_out = rec
                    abu_list.append(abu)
                    out_list.append(img_out)
            else:
                if head_layer is not None:
                    img_out = head_layer(output)
                    # normalize the final output with sigmoid or tanh function
                    img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                    out_list.append(img_out)

            index += 1

        return  out_list, abu_list
    
    def forward(self, data):
        if self.isEval:
            input_id = data
        else:
            input_id = data['img_id']  # [B]
            
        batch_size = input_id.shape[0]

        output_list, abu_list = self.forward_impl(input_id)  # a list containing [B or 2B, 3, H, W]

        if self.loss and self.training:
            b, c, h, w = output_list[-1].shape
            # NO USE
            grad_loss = 0.0
            return {
                "loss": grad_loss * self.loss_w,
                "output_list": output_list,
                "abu":abu_list,
            }
        
        return output_list, abu_list

# dim_list = [] cfg['act']
class WGSAttension(nn.Module):
    def __init__(self, dim_list, isact):
        super().__init__()
        # self.mlp = NeRV_MLP(dim_list=dim_list, act=isact)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim_list[0], dim_list[1], 1, 1, 0),
            nn.GELU(),
        )
        
    def forward(self, input1, input2):
        cres = torch.cat([input1, input2], dim=1)
        aa = self.mlp(cres)
        g = torch.sigmoid(aa)
        result = g * input1 + (1-g)*input2
        return result

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        
        return target_tensor.expand_as(input)

    def __call__(self, input, gamm=1e-7):#1 128 16 16
        input = torch.sum(input, 1)#1 16 16
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)
        return loss*gamm

class AutoEncoder(nn.Module):
    def __init__(self, L, lowerwidth, isSoftMax, dataname, isEval=False):
        super(AutoEncoder, self).__init__()
        self.dataname = dataname
        self.isSoftMax = isSoftMax
        ngf = lowerwidth#196,96
        L = lowerwidth
        self.isEval = isEval

        self.softmax = nn.Softmax(dim=1)

        layers = []
        layers.append(nn.Conv2d(L, int(ngf/2), 1, 1, 0))
        layers.append(nn.LeakyReLU(0.2, False))

        layers.append(nn.Conv2d(int(ngf/2), int(ngf/4), 1, 1, 0))
        layers.append(nn.LeakyReLU(0.2, False))

        layers.append(nn.Conv2d(int(ngf/4), int(ngf/8), 1, 1, 0))
        layers.append(nn.LeakyReLU(0.2, False))

        if self.dataname == 'pavia' or self.dataname == 'chikusei':
            layers.append(nn.Conv2d(int(ngf/8), 9, 1, 1, 0))
            layers.append(nn.LeakyReLU(0.2, False))

        self.encoder = nn.Sequential(
            *layers
        )
        
        if self.dataname == 'pavia' or self.dataname == 'chikusei':
            self.decoder = nn.Sequential(
                nn.Conv2d(9, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(int(ngf/8), 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
            )

        # self.decoder = nn.Sequential(
        #     # 非线性
        #     nn.Conv2d(int(ngf/4), int(ngf/8), kernel_size=1, stride=1, padding=0),
        #     nn.LeakyReLU(0.2, False),
        #     nn.Conv2d(int(ngf/8), 1, kernel_size=1, stride=1, padding=0),
        #     # nn.LeakyReLU(0.2, False),
        # )
            
    def forward(self,x):
        if self.isSoftMax is False:
            abu_est1 = self.encoder(x).clamp_(0,1)
        else:
            # print('softmax')
            abu_est1 = self.encoder(x)
            abu_est1 = self.softmax(abu_est1)

        re_result1 = self.decoder(abu_est1)

        if self.isSoftMax is False:
            return abu_est1, re_result1
        else:
            if self.isEval:
                #eval时添加一个全0的tensor，防止量化框架报错，并不影响eval结果
                return torch.zeros_like(abu_est1), re_result1
            else:
                return [], re_result1