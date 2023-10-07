import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import scipy.io as sio
import numpy as np
import mat73
import os
import argparse

from models.HandNet import CSI_AutoEncoder
from utils.statics import mpjpe_pck, IoU, mPA
from utils.sampling import cusDataset, split_dataset

def inference(model, testloader, device, saved = False, folder = ""):
    pred_joints_mult_2d = []
    pred_joints_mult_3d = []
    gt_joints_mult = []

    gt_mask_mult = []
    pred_mask_mult = []

    total_mpa = 0
    total_iou = 0
    total_pck = 0
    total_mpjpe = 0
    for batch_idx, ((joints,image), csi)  in enumerate(testloader):  
        joint = joints[:,:,0:21].to(device,dtype=torch.float)
        img=image.to(device,dtype=torch.float)
        csi=csi.to(device,dtype=torch.float)
        joint2d = joint[:,0:2,:] #(x,y) 200,2,21
        joint2d = joint2d.view(-1,42)
        joint3d = joint[:,2,:] # z: torch.Size([200, 21]) 
        
        with torch.no_grad():
            _, mask, twod, threed = model(csi)

        mask = torch.squeeze(mask,1) 

        # jac = jaccard(img,mask) 
        IoUerr = IoU(img,mask) 
        mPAerr = mPA(img,mask)
        mpjpe, pck = mpjpe_pck(joint2d,joint3d, twod, threed)
        total_mpa += mPAerr
        total_iou += IoUerr
        total_pck += pck
        total_mpjpe += mpjpe

        if saved:
            pred_joints_mult_2d.append(twod)
            pred_joints_mult_3d.append(threed)
            gt_joints_mult.append(joints)

            pred_mask_mult.append(mask)
            gt_mask_mult.append(image)

        torch.cuda.empty_cache()
        print(  f'{batch_idx} => mPA: {mPAerr:.3f} | => IoU: {IoUerr:.3f} | => mpjpe: {mpjpe:.3f} | =>pck: {pck:.3f}\n')

    # non_scale to scale factor is 355
    scale_factor = 355
    print(f'Total: => mPA: {total_mpa/(batch_idx+1):.3f} | => IoU: {total_iou/(batch_idx+1):.3f}' 
          f'=> mpjpe: {total_mpjpe/(batch_idx+1):.3f} (physical=>{total_mpjpe*scale_factor/(batch_idx+1):.3f} (mm)) | =>pck: {total_pck/(batch_idx+1):.3f}\n'
            )
    
    if saved:
        torch.save(pred_mask_mult,f'./{folder}/pred_mask.pt')
        torch.save(gt_mask_mult,f'./{folder}/gt_mask.pt')

        torch.save(gt_joints_mult,f'./{folder}/gt_joints.pt')
        torch.save(pred_joints_mult_2d,f'./{folder}/pred_joints_2d.pt')
        torch.save(pred_joints_mult_3d,f'./{folder}/pred_joints_3d.pt')

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default="",
                        help="number of rounds of training")
    parser.add_argument('--bs', type=int, default=80,
                        help="local batch size")
    parser.add_argument('--dir', type=str, default="/home/jisijie/new_hand/testSep/dataset",
                        help='dataset directory')
    parser.add_argument('--save', type=int, default=0,
                        help="save inference data (1) or not (0)")

    args = parser.parse_args()
    return args

# Set CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

args = args_parser()
folder = args.folder
BATCH_SIZE = args.bs
dir = args.dir

# model setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CSI_AutoEncoder()
model.load_state_dict(torch.load(f'./{folder}/handnet_runexp1.pth',map_location=device), strict=False)
model.to(device)

# data loading rawdata
csi_data = mat73.loadmat(f'{dir}/csi_data.mat')['csi_data']
joints_data = sio.loadmat(f'{dir}/joints_data.mat')['joints_data']
images_data = sio.loadmat(f'{dir}/image_data.mat')['image_data']
images_data = images_data.astype(np.float32)

testset = cusDataset(csi_data, images_data, joints_data)
test_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True)

inference(model, test_loader, device, saved=args.save, folder=folder)
