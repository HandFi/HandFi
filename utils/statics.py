import torch
#from packaging import version
__all__ = ['AverageMeter', 'evaluator', 'mpjpe_pck', 'IoU', 'mPA']
from scipy.spatial import distance


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def mpjpe_pck(gt2d,gt3d, pre2d, pre3d):
    with torch.no_grad(): 
        
        dist = gt2d - pre2d
        dist = dist * dist
        distz = gt3d - pre3d
        difz = distz * distz
        difx = dist[:,0:21]
        dify = dist[:,21:42]
        summm = difx+dify+difz
        jlist = torch.sqrt(summm)
        bs = gt2d.size()[0]
        batchmpjpe = jlist.mean()
        pck=[]
        for i in range(bs):
            temp=torch.where(jlist[i]<0.05) 
            pck.append(temp[0].size(0) / jlist[i].size(0))
            # import pdb; pdb.set_trace()
        batchpck = sum(pck)/bs
        return batchmpjpe, batchpck

def IoU(ori, dec):  # ori = gt dec = pred
   # with torch.no_grad():
    ori = torch.round(ori)
    dec= torch.round(dec)
    overlap = ori * dec # AND Gate
    union = (ori + dec) > 0 # OR Gate
    
    iou =  torch.sum(overlap,(1,2)) / torch.sum(union,(1,2))
    return iou.mean()

def jaccard(y_true, y_pred):
    #
    with torch.no_grad():
        y_pred = torch.round(y_pred)
        temp = y_true * y_pred
        intersection = torch.sum(temp,(1,2))
        union = torch.sum(y_true,(1,2)) + torch.sum(y_pred,(1,2)) - intersection
        result = intersection/union
        return result.mean()

def mPA(ori, dec):
    with torch.no_grad():
        dec = torch.round(dec)
        dec = torch.squeeze(dec,1)
        falsePred = torch.sum(torch.logical_xor(ori,dec),(1,2))
    #bug.
        temp = falsePred.cpu().detach().numpy() / 12996
        result = 1 - temp
        result =torch.tensor(result)
    return result.mean()
   

def mpjpe(gt,pre):
    with torch.no_grad(): 
        
        dist = gt - pre
        dist = dist * dist
        difx = dist[:,0:21]
        dify = dist[:,21:42]
        summm = difx+dify
        jlist = torch.sqrt(summm)
        bs = gt.size()[0]
        batchmpjpe = jlist.mean()
        pck=[]
        for i in range(bs):

            temp=torch.where(jlist[i]<0.05) 
            pck.append(temp[0].size(0))
        return batchmpjpe,sum(pck)/bs

def pck(gt,pre):
    with torch.no_grad(): 
        #import pdb;pdb.set_trace()
        #for z data
        dist = gt - pre
        dist = dist * dist
        #difx = dist[:,0:21]
        bs = gt.size()[0]
        #dify = dist[:,21:42]
        #summm = difx+dify
        zlist = torch.sqrt(dist)
        #result= list.mean(1)
        #import pdb;pdb.set_trace()
        batchmpjpe = zlist.mean()
        pck=[]
        for i in range(bs):

            temp=torch.where(zlist[i]<0.05) 
            pck.append(temp[0].size(0))
        #A = torch.dist(gt,pre,p=1)/gt.size(0)
        return batchmpjpe,sum(pck)/bs

def pck3D(gt,pre):
    with torch.no_grad(): 
        #import pdb;pdb.set_trace()
        #for z data
        dist = gt - pre
        dist = dist * dist
        #difx = dist[:,0:21]
        bs = gt.size()[0]
        #dify = dist[:,21:42]
        #summm = difx+dify
        zlist = torch.sqrt(dist)
        #result= list.mean(1)
        #import pdb;pdb.set_trace()
        batchmpjpe = zlist.mean()
        # pck=[]
        # for i in range(bs):

        #     temp=torch.where(zlist[i]<0.05) 
        #     pck.append(temp[0].size(0))
        #A = torch.dist(gt,pre,p=1)/gt.size(0)
        return batchmpjpe

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1



