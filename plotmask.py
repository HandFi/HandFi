import torch
import cv2
import argparse
import time

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', nargs='?', type=int, default=0)
    parser.add_argument('--j', nargs='?', type=int, default=0)
    parser.add_argument('--folder', nargs='?', type=str, default=".")
    parser.add_argument('--exp', nargs='?', type=str, default=".")


    args = parser.parse_args()
    return args

args = args_parser()

k = args.k
j = args.j

gt = torch.load(f'./{args.folder}/gt_mask.pt', map_location=torch.device('cpu'))
pre = torch.load(f'./{args.folder}/pred_mask.pt', map_location=torch.device('cpu'))
# import pdb; pdb.set_trace()
gt = torch.round(gt[k][j]).cpu().detach().numpy()
pre = torch.round(pre[k][j]).cpu().detach().numpy()

cv2.imwrite(f'./{args.folder}/{args.exp}/maskgt{k}_{j}_{str(time.time())[3:9]}.jpg', gt*255)
cv2.imwrite(f'./{args.folder}/{args.exp}/maskpre{k}_{j}_{str(time.time())[3:9]}.jpg', pre*255)

