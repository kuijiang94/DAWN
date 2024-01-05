import torch
import torch.nn as nn
from subpixel import shuffle_down, shuffle_up###################
import torch.nn.functional as F
from pdb import set_trace as stx
from Blanced_attention import BlancedAttention, BlancedAttention_CAM_SAM_ADD
from coordatt import CoordAtt
from torchvision import transforms as trans
#import numpy as np
#from wavelet import wt,iwt
#from pywt import dwt2, wavedec2
#from lifting import WaveletHaar2D, LiftingScheme2D, Wavelet2D
##########################################################################
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return [x_LL, x_HL, x_LH, x_HH]#torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width]) #[1, 12, 56, 56]
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r**2)), r * in_height, r * in_width
    # print(out_batch, out_channel, out_height, out_width) #1 3 112 112
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    # print(x1.shape) #torch.Size([1, 3, 56, 56])
    # print(x2.shape) #torch.Size([1, 3, 56, 56])
    # print(x3.shape) #torch.Size([1, 3, 56, 56])
    # print(x4.shape) #torch.Size([1, 3, 56, 56])
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


# 
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 

    def forward(self, x):
        return dwt_init(x)


# 
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
##########################################################################
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def st_conv(in_channels, out_channels, kernel_size, bias=False, stride = 2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)		
##########################################################################
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
##########################################################################
## S2FB
class S2FB_2(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_2, self).__init__()
        self.DSC = depthwise_separable_conv(n_feat*2, n_feat)
        #self.CON_FEA = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2):
        FEA_1 = self.DSC(torch.cat((x1,x2), 1))
        #FEA_2 = self.CON_FEA(torch.cat((x1,x2), 1))
        #resin = FEA_1 + FEA_2
        res= self.CA_fea(FEA_1) + x1
        #res += resin
        return res#x1 + resin
##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
		
# contrast-aware channel attention module
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
	
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
##########################################################################
class h_sigmoid(nn.Module):
    #def __init__(self, inplace=True):
    def __init__(self, inplace=False):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
############  V (H) attention Layer
class CoordAtt_V(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_V, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))#nn.AdaptiveAvgPool2d((None, 1)),for training

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h_pool = self.pool_h(x)##n,c,h,1
        y = self.conv1(x_h_pool)
        y_bn = self.bn1(y)
        y_act = self.act(y_bn) 
		
        a_h_att = self.conv_h(y_act).sigmoid()


        out = identity * a_h_att

        return out
############  H (W) attention Layer
class CoordAtt_H(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_H, self).__init__()
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        #x_h = self.pool_h(x)
        x_w_pool = self.pool_w(x).permute(0, 1, 3, 2)##n,c,W,1
        y = self.conv1(x_w_pool)
        y_bn = self.bn1(y)
        y_act = self.act(y_bn) 
        
        y_act_per = y_act.permute(0, 1, 3, 2)
        a_w_att = self.conv_w(y_act_per).sigmoid()

        out = identity * a_w_att

        return out
##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        #self.CA = CoordAtt(n_feat, n_feat)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = res + x
        return res
		
## Enhanced Channel Attention Block (ECAB)
class ECAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ECAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        #self.CA = CoordAtt(n_feat, n_feat)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res
		
## Channel Attention Block (CAB)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = []
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))
        modules_body.append(act)
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        #self.CA = CoordAtt(n_feat, n_feat)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res

##########################################################################
## Supervised Attention Module
# class SAM(nn.Module):
    # def __init__(self, n_feat, kernel_size, bias):
        # super(SAM, self).__init__()
        # self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        # self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        # self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    # def forward(self, x, x_img):
        # x1 = self.conv1(x)
        # img = self.conv2(x) + x_img
        # x2 = torch.sigmoid(self.conv3(img))
        # x1 = x1*x2
        # x1 = x1+x
        # return x1, img
##########################################################################
##########################################################################
## Direction Attention Block (DAB)
class DAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(DAB, self).__init__()

        self.Main_fea = nn.Sequential(CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.V_fea = nn.Sequential(CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.H_fea = nn.Sequential(CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act))

        self.V_ATT = CoordAtt_V(n_feat, n_feat)
        self.H_ATT = CoordAtt_H(n_feat, n_feat)


    def forward(self, x):#[x_main_rain, x_V_fea, x_H_fea]
        main_fea = self.Main_fea(x[0])
        v_ATT_fea = self.V_ATT(main_fea)
        h_ATT_fea = self.H_ATT(main_fea)
		
        v_feas = self.V_fea(x[1]) - v_ATT_fea 
        h_feas = self.H_fea(x[2]) - h_ATT_fea
        return [main_fea, v_feas, h_feas]
		
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [DAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        #modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):#[x_main_rain, x_V_fea, x_H_fea]
        # x_main = x[0]
        # x_V = x[1]
        # x_H = x[2]
        res = self.body(x)
        #res = res + x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORSNet, self).__init__()
        act=nn.PReLU()
		
        self.shallow_feat1 = nn.Sequential(conv(3*4, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
		
        self.orb = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
		
        self.tail_main_LL = conv(n_feat, 3, kernel_size, bias=bias)
        self.tail_main_HH = conv(n_feat, 3, kernel_size, bias=bias)
        self.tail_V_fea = conv(n_feat, 3, kernel_size, bias=bias)
        self.tail_H_fea = conv(n_feat, 3, kernel_size, bias=bias)
		
    def forward(self, x):#x_LL, x_HL, x_LH, x_HH
        # H = x.size(2)
        # W = x.size(3)
		############ WaveletHaar2D, DWT
        x_LL = x[0]
        x_V = x[1]
        x_H = x[2]
        x_HH = x[3]
		############ wt
        # x_LL = x[:,0:3,:,:]#x[0]
        # x_V = x[:,3:6,:,:]#x[1]##Vertical
        # x_H = x[:,6:9,:,:]#x[2]##Horizontal
        # x_HH = x[:,9:12,:,:]#x[3]
		
        x_main_rain = self.shallow_feat1(torch.cat([x_LL, x_V, x_H, x_HH],1))
        x_V_fea = self.shallow_feat2(x_V)
        x_H_fea = self.shallow_feat3(x_H)
		
        x_out= self.orb([x_main_rain, x_V_fea, x_H_fea])
		
        #x_main = self.tail_main_rain(x_out[0])
        x_LL_rain = self.tail_main_LL(x_out[0])
        x_HH_rain = self.tail_main_HH(x_out[0])
        #x_LL_rain, x_HH_rain = torch.split(x_main, (3, 3), dim=1)

        x_V_img = self.tail_V_fea(x_out[1])
        x_H_img = self.tail_H_fea(x_out[2])
		
        x_cat = torch.cat((x_LL-x_LL_rain, x_V_img, x_H_img, x_HH-x_HH_rain), 1)
        return x_cat

##########################################################################
class DAWN(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64, kernel_size=3, reduction=4, num_cab=25, bias=False):
        super(DAWN, self).__init__()

        act=nn.PReLU()
        #self.wavelet1 = WaveletHaar2D()
        self.dwt = DWT()
        self.orsnet = ORSNet(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.iwt = IWT()
		
    def forward(self, img): #####b,c,h,w
        #print(x_img.shape)
        #wave_out1 = wt(img)
        dwt_fea = self.dwt(img)
        #print(dwt_fea)
        #wave_out1 = self.wavelet1(img)
        orsnet_out = self.orsnet(dwt_fea)
        iwt_fea = self.iwt(orsnet_out)#iwt(orsnet_out)
        return iwt_fea#, orsnet_out[0], orsnet_out[0], orsnet_out[0]
