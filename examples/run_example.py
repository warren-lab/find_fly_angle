import sys
import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
# from find_fly_angle import find_fly_angle
from find_fly_angle.find_fly_anglev2 import find_fly_angle

import time
=======
from find_fly_angle import find_fly_angle
import pdb
#import matplotlib



>>>>>>> d7adfc9ca7dca752b90ee39b52ff0cf6535c466e

image_dir = sys.argv[1]

threshold = 25
mask_scale = 0.9

# Get test images
image_files = os.listdir(image_dir)
image_files = [f for f in image_files if os.path.splitext(f)[1]=='.png']
image_files = [f for f in image_files if not os.path.isfile(f)]
image_files.sort()

frame_list = []
angle_list = []
discrim_list=[]
perimeter_discrim_list=[]
perim_1_list=[]
perim_0_list=[]
moment_1_list=[]
moment_0_list=[]

<<<<<<< HEAD
for file_number, file_name in enumerate(image_files[1670:]):

    image = cv2.imread(os.path.join(image_dir,file_name),cv2.IMREAD_GRAYSCALE)
    print(np.shape(image))
    angle, angle_data = find_fly_angle(image, threshold, mask_scale)

    frame_list.append(file_number)
    angle_list.append(angle)
    # show the filename
    cv2.putText(angle_data['contour_image'],file_name,(20,20),1,1,(255,255,255),1)
=======
for file_number, file_name in enumerate(image_files[0:300]):

    image = cv2.imread(os.path.join(image_dir,file_name),cv2.IMREAD_GRAYSCALE)
    
    angle, angle_data, orient_dict = find_fly_angle(image, threshold, mask_scale)

    frame_list.append(file_number)
    angle_list.append(angle)
    discrim_list.append(orient_dict['discrim'])
    perimeter_discrim_list.append(orient_dict['perimeter_discrim'])
    perim_1_list.append(orient_dict['perimeter_1'])
    perim_0_list.append(orient_dict['perimeter_0'])
    moment_0_list.append(orient_dict['moment_0'])
    moment_1_list.append(orient_dict['moment_1'])
    current_image0=orient_dict['image_0']
    current_image1=orient_dict['image_1']


    # create a png that shows the two halves...


    # save moments

>>>>>>> d7adfc9ca7dca752b90ee39b52ff0cf6535c466e
    cv2.imshow('contour image', angle_data['contour_image'])
    cv2.imshow('rotated image', angle_data['rotated_image'])

    #cv2.imshow('threshold image', angle_data['threshold_image'])
    #cv2.imshow('shifted image', angle_data['shifted_image'])
    #cv2.imshow('shifted_threshold image', angle_data['shifted_threshold_image'])
    #cv2.imshow('rotated_threshold image', angle_data['rotated_threshold_image'])

    print(('{0}/{1}: {2}, angle: {3:1.2f}'.format(file_number, len(image_files), file_name,np.rad2deg(angle))))

    time.sleep(1)

    cv2.waitKey(50)

cv2.destroyAllWindows()

<<<<<<< HEAD
angle_list = list(map(np.rad2deg, angle_list))
plt.plot(frame_list, angle_list,'.')
plt.xlabel('frame (#)')
=======
>>>>>>> d7adfc9ca7dca752b90ee39b52ff0cf6535c466e

#angle_list = map(np.rad2deg, angle_list)
#plt.plot(frame_list, angle_list,'.')
#plt.xlabel('frame (#)')

#plt.ylabel('angle (deg)')
#plt.grid('on')
#plt.show()
plt.ion()
plt.figure()
plt.scatter(frame_list,discrim_list)


