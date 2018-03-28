import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from find_fly_angle import find_fly_angle

threshold = 50
mask_scale = 0.9

image_dir = 'sample_images'
max_frame = 551

# Get test images
image_files = os.listdir(image_dir)
image_files = [f for f in image_files if os.path.splitext(f)[1]=='.png']
image_files = [f for f in image_files if not os.path.isfile(f)]
image_files.sort()

frame_list = []
angle_list = []

for file_number, file_name in enumerate(image_files):

    if file_number >= max_frame:
        break

    image = cv2.imread(os.path.join(image_dir,file_name),cv2.IMREAD_GRAYSCALE)
    angle, angle_data = find_fly_angle(image, threshold, mask_scale)

    frame_list.append(file_number)
    angle_list.append(angle)

    cv2.imshow('contour image', angle_data['contour_image'])
    cv2.imshow('rotated image', angle_data['rotated_image'])

    #cv2.imshow('threshold image', angle_data['threshold_image'])
    #cv2.imshow('shifted image', angle_data['shifted_image'])
    #cv2.imshow('shifted_threshold image', angle_data['shifted_threshold_image'])
    #cv2.imshow('rotated_threshold image', angle_data['rotated_threshold_image'])

    print('{0}/{1}: {2}, angle: {3:1.2f}'.format(file_number, len(image_files), file_name,np.rad2deg(angle)))

    cv2.waitKey(20)

cv2.destroyAllWindows()

angle_list = map(np.rad2deg, angle_list)
plt.plot(frame_list, angle_list)
plt.xlabel('frame (#)')

plt.ylabel('angle (deg)')
plt.grid('on')
plt.show()

