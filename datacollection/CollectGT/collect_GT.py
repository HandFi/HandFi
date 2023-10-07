from datetime import datetime
import imp
import os, sys, inspect
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = './lib/x64' if sys.maxsize > 2**32 else './lib/x86'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import time
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, image
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
import ctypes
import cv2 as cv

import Leap
import pdb
import scipy.io as sio

# Leap Motion Controller Setup
controller = Leap.Controller()
controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
NUM_POINTS = 22

SAVE = False
SAVE = True
points_list = []

start_time = time.time()
'''
finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
bone_names = ['MCP', 'PIP', 'DIP', 'TIP']
# We can of course generate column names on the fly:
for finger in finger_names:
	for bone in bone_names:
		for dim in ["x","y","z"]:
			columns.append(f"{finger}_{bone}_{dim}")
print(columns)
'''
columns = [
		"Palm_x", "Palm_y", "Palm_z",
		'Thumb_MCP_x', 'Thumb_MCP_y', 'Thumb_MCP_z',
		'Thumb_PIP_x', 'Thumb_PIP_y', 'Thumb_PIP_z',
		'Thumb_DIP_x', 'Thumb_DIP_y', 'Thumb_DIP_z',
		'Thumb_TIP_x', 'Thumb_TIP_y', 'Thumb_TIP_z',
		'Index_MCP_x', 'Index_MCP_y', 'Index_MCP_z',
		'Index_PIP_x', 'Index_PIP_y', 'Index_PIP_z',
		'Index_DIP_x', 'Index_DIP_y', 'Index_DIP_z',
		'Index_TIP_x', 'Index_TIP_y', 'Index_TIP_z',
		'Middle_MCP_x', 'Middle_MCP_y', 'Middle_MCP_z',
		'Middle_PIP_x', 'Middle_PIP_y', 'Middle_PIP_z',
		'Middle_DIP_x', 'Middle_DIP_y', 'Middle_DIP_z',
		'Middle_TIP_x', 'Middle_TIP_y', 'Middle_TIP_z',
		'Ring_MCP_x', 'Ring_MCP_y', 'Ring_MCP_z',
		'Ring_PIP_x', 'Ring_PIP_y', 'Ring_PIP_z',
		'Ring_DIP_x', 'Ring_DIP_y', 'Ring_DIP_z',
		'Ring_TIP_x', 'Ring_TIP_y', 'Ring_TIP_z',
		'Pinky_MCP_x', 'Pinky_MCP_y', 'Pinky_MCP_z',
		'Pinky_PIP_x', 'Pinky_PIP_y', 'Pinky_PIP_z',
		'Pinky_DIP_x', 'Pinky_DIP_y', 'Pinky_DIP_z',
		'Pinky_TIP_x', 'Pinky_TIP_y', 'Pinky_TIP_z',
		"Wrist_x", "Wrist_y", "Wrist_z"
		]
# Convert this to headers for numpy saving...
headers = ""
for col in columns:
	headers+= col
	headers+= ","
headers = headers[:-2]

label_input = "u"
numshands = 40

def on_close(event):
	print("Closed Figure")

	end_time = time.time()
	# print(f"Time elapsed: {end_time - start_time}")
	# print(f"Len points_list: {len(points_list)}")

	if (SAVE):
		print("Saving all points gathered")
		now = datetime.now()
		now.strftime("%Y-%m-%d-%H-%M-%S")
		# ave hands
		aveList = []
		tempList = []
		for i in range(numshands * 3):
			if i%3 == 0:
				tempList = points_list[i]
			else:
				tempList += points_list[i]
				if i%3 == 2:
					tempList /= 3
					aveList.append(tempList)			

		# Alternatively use pandas to remove need to make headers string.
		np.savetxt("./dataset/" + label_input.upper() + "/label" + label_input.upper() + str(now.strftime("%Y-%m-%d-%H-%M-%S")) + ".csv", aveList, delimiter=',', header=headers, comments='')
		# np.savetxt("./datasetRange/" + label_input.upper() + str(now.strftime("%Y-%m-%d-%H-%M-%S")) + ".csv", points_list, delimiter=',', header=headers, comments='')
		

def convert_distortion_maps(image):
    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length / 2, dtype=np.float32)
    ymap = np.zeros(distortion_length / 2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length / 2 - i / 2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length / 2 - i / 2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width / 2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width / 2))

    # resize the distortion map to equal desired destination image size
    resized_xmap = cv.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv.INTER_LINEAR)
    resized_ymap = cv.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv.INTER_LINEAR)

    # Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv.CV_32FC1,
                                                                 nninterpolation=False)

    return coordinate_map, interpolation_coefficients


def undistort(image, coordinate_map, coefficient_map, width, height): 
    destination = np.empty((width, height), dtype=np.ubyte)

    # wrap image data in numpy array
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    # remap image to destination
    destination = cv.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation=cv.INTER_LINEAR)

    # resize output to desired destination size
    destination = cv.resize(destination,
                             (width, height),
                             0, 0,
                             cv.INTER_LINEAR)
    return destination


# Matplotlib Setup
fig = plt.figure()
fig.canvas.mpl_connect('close_event', on_close)
ax = fig.add_subplot(111, projection='3d', xlim=(-300, 300), ylim=(-200, 400), zlim=(-300, 300))
ax.view_init(elev=45., azim=122)

points = np.zeros((3, NUM_POINTS))
patches = ax.scatter(points[0], points[1], points[2], s=[20]*NUM_POINTS, alpha=1)

def get_points(mask):
	frame = controller.frame()
	
	image = frame.images[0]
	maps_initialized = False
	flag = True
	if image.is_valid and mask:
		print(mask)
		if not maps_initialized:
			# left_coordinates, left_coefficients = convert_distortion_maps(frame.images[0])
			right_coordinates, right_coefficients = convert_distortion_maps(frame.images[1])

			maps_initialized = True

		# undistorted_left = undistort(image, left_coordinates, left_coefficients, 600, 600)
		undistorted_right = undistort(frame.images[1], right_coordinates, right_coefficients, 600, 600)
		# undistorted_left = np.reshape(undistorted_left, (600, 600))
		# (thresh, im_bw) = cv.threshold(undistorted_left, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
		(thresh, im_bw) = cv.threshold(undistorted_right, 150, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
		# display images
		# cv.imshow('Left Camera', im_bw)
		now = datetime.now()	
		cv.imwrite("./dataset/" + label_input.upper() + "/label" + label_input.upper() + str(now.strftime("%Y-%m-%d-%H-%M-%S-%f")) + ".jpg", im_bw[200:400, 230:430])
	
	hand = frame.hands.rightmost
	if not hand.is_valid: return np.array(patches._offsets3d)
	fingers = hand.fingers

	X = []
	Y = []
	Z = []

	# Add the position of the palms
	X.append(-1 *hand.palm_position.x)
	Y.append(hand.palm_position.y)
	Z.append(hand.palm_position.z)

	# Add Elbow
	# arm = hand.arm
	# X.append(arm.elbow_position.x)
	# Y.append(arm.elbow_position.y)
	# Z.append(arm.elbow_position.z)

	# Add fingers
	for finger in fingers:
		for b in range(0, 4):
			'''
			0 = JOINT_MCP The metacarpophalangeal joint, or knuckle, of the finger.
			1 = JOINT_PIP The proximal interphalangeal joint of the finger. This joint is the middle joint of a finger.
			2 = JOINT_DIP The distal interphalangeal joint of the finger. This joint is closest to the tip.
			3 = JOINT_TIP The tip of the finger.
			'''
			bone = finger.bone(b)
			X.append(-1 * bone.next_joint[0])
			Y.append(bone.next_joint[1])
			Z.append(bone.next_joint[2])
	# Add wrist position
	X.append(-1 * hand.wrist_position.x)
	Y.append(hand.wrist_position.y)
	Z.append(hand.wrist_position.z)
	return np.array([X, Z, Y])

def save_points(points,name='points.csv'):
	# Save one single row/frame to disk
	np.savetxt(name, points, delimiter=',')

def animate(i):
	points = get_points()
	if (SAVE):
		points_list.append(points.flatten())

	return patches,


def main():	
	time.sleep(0.5)
	for i in range(numshands*3): # one sec 10 times
		start=time.time()
		if i % 3 == 1:
			points = get_points(True)
		else:
			points = get_points(False)
		print(i)
		if (SAVE):
			points_list.append(points.flatten())
			# if i == 0:
			# 	image_list = img
			# # elif i == 1:
			# # 	# image_list = np.dstack([image_list,img])
			# # 	image_list = np.dstack((image_list,img))
			# # 	# pdb.set_trace()
			# else:
			# 	image_list = np.dstack((image_list,img))
			# 	# np.append(image_list, np.atleast_3d(img), axis=2)
			# 	# pdb.set_trace()
		# print(time.time()-start)
		while ((time.time()-start)<0.19):
			pass
	# now = datetime.now()
	# now.strftime("%Y-%m-%d-%H-%M-%S")
	# sio.savemat("./datasetRec/" + label_input.upper() + str(now.strftime("%Y-%m-%d-%H-%M-%S")) + ".mat", {'img_list': image_list})
	

if __name__ == '__main__':
	main()