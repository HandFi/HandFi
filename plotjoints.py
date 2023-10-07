import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import scipy.io as sio
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

gt = torch.load(f'./{args.folder}/gt_joints.pt', map_location=torch.device('cpu'))
gt = gt[k][j].cpu().detach().numpy()

pre = torch.load(f'./{args.folder}/pred_joints_2d.pt', map_location=torch.device('cpu'))
prez = torch.load(f'./{args.folder}/pred_joints_3d.pt', map_location=torch.device('cpu'))

pre = pre[k][j].view(-1,2,21)
prez = prez[k][j]

pre = pre[0][:][:].cpu().detach().numpy()
prez = prez.cpu().detach().numpy()

pre = np.vstack((pre,prez))

xs, xe = 0.1,0.7
ys, ye = 0,0.6
zs, ze = 0.2,0.8

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))

def plot_lines(points):
	mcps = []

	# Wrist
	wrist = points[:,0]
	colormap = ['#40E0D0', '#FF7F50', '#DFFF00', '#CCCCFF', '#9FE2BF', '#6495ED']
	# For Each of the 5 fingers
	for i in range(0,5):
		n = 4*i + 1

		# Get each of the bones
		mcp = points[:,n+0]
		pip = points[:,n+1]
		dip = points[:,n+2]
		tip = points[:,n+3]

		# Connect the lowest joint to the middle joint
		bot = plt3d.art3d.Line3D([mcp[0], pip[0]], [mcp[1], pip[1]], [mcp[2], pip[2]], color=colormap[i], linewidth=4)
		ax.add_line(bot)

		# Connect the middle joint to the top joint
		mid = plt3d.art3d.Line3D([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]], color=colormap[i], linewidth=4)
		ax.add_line(mid)

		# Connect the top joint to the tip of the finger
		top = plt3d.art3d.Line3D([dip[0], tip[0]], [dip[1], tip[1]], [dip[2], tip[2]], color=colormap[i], linewidth=4)
		ax.add_line(top)

		# Connect each of the fingers together
		mcps.append(mcp)
	for mcp in range(0,4):
		line = plt3d.art3d.Line3D([mcps[mcp][0], mcps[mcp+1][0]],
									[mcps[mcp][1], mcps[mcp+1][1]],
									[mcps[mcp][2], mcps[mcp+1][2]],color=colormap[5], linewidth=4)
		ax.add_line(line)
	# Create the right side of the hand joining the pinkie mcp to the "wrist"
	line = plt3d.art3d.Line3D([wrist[0], mcps[4][0]],
									[wrist[1], mcps[3+1][1]],
									[wrist[2], mcps[3+1][2]],color=colormap[5], linewidth=4)
	ax.add_line(line)

	# Generate the "Wrist", note right side is not right.
	line = plt3d.art3d.Line3D([wrist[0], mcps[0][0]],
									[wrist[1], mcps[0][1]],
									[wrist[2], mcps[0][2]],color=colormap[5], linewidth=4)
	ax.add_line(line)

	# palmx = (1 * (points[0][0] + points[0][1]) / 2 + 3 * (points[0][9] + points[0][13]) / 2) / 4
	# palmy = (1 * (points[1][0] + points[1][1]) / 2  + 3 * (points[1][9] + points[1][13]) / 2) / 4
	# palmz = (1 * (points[2][0] + points[2][1]) / 2  + 3 * (points[2][9] + points[2][13]) / 2) / 4
	ax.scatter(points[0,0], points[1,0], points[2,0], c = 'w', s=[70], marker='o', edgecolors= 'w', alpha=1)
	
	# palmx = points[0][0]
	# palmx = points[0][0]

def axsetup(ax):
    # Reset the plot
    ax.cla()
    # Really you can just update the lines to avoid this
    # ax.view_init(elev=55, azim=78)
    ax.view_init(elev=60, azim=122)
    ax.set_xlim3d([xs, xe])
    ax.set_xlabel('')
    ax.set_ylim3d([ys, ye])
    ax.set_ylabel('')
    ax.set_zlim3d([zs, ze])
    ax.set_zlabel('')
    fig.set_facecolor('black')
    ax.set_facecolor('black') 
    ax.set_axis_off()

    # bottom
    line = plt3d.art3d.Line3D([xs, xe],
                            [ys , ys],
                            [zs, zs],color='#808080', linewidth=2)
    ax.add_line(line)

    line = plt3d.art3d.Line3D([xs, xe],
                            [ye, ye],
                            [zs, zs],color='#808080', linewidth=2)
    ax.add_line(line)

    line = plt3d.art3d.Line3D([xs, xs],
                            [ys, ye],
                            [zs, zs],color='#808080', linewidth=2)
    ax.add_line(line)

    line = plt3d.art3d.Line3D([xe, xe],
                            [ys, ye],
                            [zs, zs],color='#808080', linewidth=2)
    ax.add_line(line)
    # top
    line = plt3d.art3d.Line3D([xs, xe],
                            [ys, ys],
                            [ze, ze],color='#808080', linewidth=2)
    ax.add_line(line)

    line = plt3d.art3d.Line3D([xe, xe],
                            [ys, ye],
                            [ze, ze],color='#808080', linewidth=2)
    ax.add_line(line)
    # middle
    line = plt3d.art3d.Line3D([xs, xs],
                            [ys, ys],
                            [zs, ze],color='#808080', linewidth=2)
    ax.add_line(line)
    line = plt3d.art3d.Line3D([xe, xe],
                            [ys, ys],
                            [zs, ze],color='#808080', linewidth=2)
    ax.add_line(line)
    line = plt3d.art3d.Line3D([xe, xe],
                            [ye, ye],
                            [zs, ze],color='#808080', linewidth=2)
    ax.add_line(line)

def save_figure(joints, file_name):
    axsetup(ax)
    patches = ax.scatter(joints[0], joints[1], joints[2], c = 'w', s=[22]*21, marker='o', edgecolors= 'w', alpha=1)
    plot_lines(joints)
    plt.savefig(file_name)

save_figure(pre, f'./{args.folder}/{args.exp}/jointspre{k}_{j}_{str(time.time())[3:9]}.png')
save_figure(gt, f'./{args.folder}/{args.exp}/jointsgt{k}_{j}_{str(time.time())[3:9]}.png')

