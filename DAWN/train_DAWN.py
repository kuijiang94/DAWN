import os

from config import Config 
opt = Config('training_DAWN.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True
from thop import profile
from SSIM import SSIM

#from loss import CharbonnierLoss, HuberLoss, GWLoss, PyramidLoss, LapPyrLoss

torch.cuda.is_available()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from DAWN import DAWN


import losses
#from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx


file1 = 'DAWN_PAR.txt'
file2 = 'DAWN_PSNR.txt'
file3 = 'DAWN_LOSS.txt'
######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR

######### Model ###########
model_restoration = DAWN()
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 0:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL
optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)


######### Scheduler ###########
warmup_epochs = 3
#scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
#scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
#scheduler.step()

############### me
#lambda1=lambda epoch:0.1**(epoch//20)
#lambda2=lambda epoch:0.95** epoch

#scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda2,last_epoch=-1)

scheduler = optim.lr_scheduler.StepLR(step_size=80, gamma=0.8, optimizer=optimizer)####step_size epoch, best_epoch 445
#scheduler.step()

######## Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, '_best.pth')
    # path_chk_rest    = utils.get_last_path(model_dir, '_epoch_90.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)
	
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_SSIM = SSIM()
criterion_fea = losses.CharbonnierLoss_dwt()

#PyLoss = losses.PyramidLoss(num_levels=3, pyr_mode='gau', loss_mode='cb', reduction='mean')
LapPyLoss = losses.LapPyrLoss(num_levels=3, lf_mode='ssim', hf_mode='cb', reduction='mean')
#criterion_WT = losses.DWT()

#train_batchsize = opt.OPTIM.BATCH_SIZE
######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
#train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size':opt.TRAINING.VAL_PS})
#val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, drop_last=False, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0


for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    SSIM_all = 0
    train_id = 1
    train_sample = 0
    #model_restoration.train()
    model_restoration.train().cuda()
	
    torch.autograd.set_detect_anomaly(True)
	
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()
        criterion_char.cuda()
        criterion_edge.cuda()
        criterion_SSIM.cuda()
        criterion_fea.cuda()
		
        restored = model_restoration(input_)
        restored_ = restored.cuda()
        #restored[0] = restored[0].cuda()
        #restored[1] = restored[1].cuda()
        #restored[2] = restored[2].cuda()
        # Compute loss at each stage
        #loss_char_V = criterion_char(restored[1],criterion_WT(target)[1])
        #loss_char_H = criterion_char(restored[2],criterion_WT(target)[2])
        #loss_char_fea = criterion_fea(restored_,target)#np.sum([criterion_char(restored[j],target) for j in range(len(restored))])
		
		
        center_idx = input_.size(1) // 2
        # loss for luminance channel
        #l_pix_y = LapPyLoss(restored_, target, restored_[:, 0:1, :, :], target[:, 0:1, :, :])
        #print(restored_[:, 0:1, :, :], target[:, 0:1, :, :].contiguous().shape)
        # loss for color channel
        #l_pix_c = LapPyLoss(restored_, target, restored_[:, 1:3, :, :], target[:, 1:3, :, :])
        #print(restored_[:, 1:3, :, :], target[:, 1:3, :, :].contiguous().shape)
        #l_pix = l_pix_y + l_pix_c
		
        loss_char = criterion_char(restored_,target)#np.sum([criterion_char(restored[j],target) for j in range(len(restored))])
        loss_edge = criterion_edge(restored_,target)#np.sum([criterion_edge(restored[j],target) for j in range(len(restored))])
        loss_SSIM = criterion_SSIM(restored_,target)#criterion_SSIM(restored[0],target)#np.sum([criterion_SSIM(restored[j],target) for j in range(len(restored))])
        loss = (0.3*loss_char) + (0.2*loss_edge) - (0.15*loss_SSIM)# + 0.05*l_pix# + (0.1*loss_char_fea)# + 0.1*loss_char_V + 0.03*loss_char_H# 0.15,0.2
		
        loss.backward()
        optimizer.step()

        epoch_loss +=loss.item()
        SSIM_all +=loss_SSIM.item()
        train_sample +=1
    SSIM = SSIM_all/train_sample
    # if epoch ==1:
        # flops, params = profile(model_restoration, (input_,))
        # print('flops: ', flops, 'params: ', params)
        # a = str("[flops: %.2f --- params %d]" % (flops, params))
        # par_file = open(file1, 'a+')
        # par_file.write(a)
        # par_file.write('\n')  
        # par_file.close()
    #### Evaluation ####
    if epoch%opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)
				
            restored = restored.cuda()

            for res,tar in zip(restored,target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
            # restored = restored.cuda()

            # for res,tar in zip(restored,target):
                # psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()
		
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))
        format_str1 = ('Epoch: %d, PSNR: %.4f, best_epoch: %d, Best_PSNR: %.4f')
        a = str(format_str1 % (epoch, psnr_val_rgb, best_epoch, best_psnr))
        PSNR_file = open(file2, 'a+')
        PSNR_file.write(a)
        PSNR_file.write('\n')
        PSNR_file.close()
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tSSIM: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss, SSIM, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    format_str = ('Epoch: %d, Time: %.4f, Loss: %.4f, SSIM: %.4f, LearningRate: %.8f')
    a = str(format_str % (epoch, time.time()-epoch_start_time, epoch_loss, SSIM, scheduler.get_lr()[0]))
    loss_file = open(file3, 'a+')
    loss_file.write(a)
    loss_file.write('\n')
    loss_file.close()
    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 

