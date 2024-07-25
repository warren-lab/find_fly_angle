#!/usr/bin/env python3

import numpy as np
import cv2


def find_fly_angle(image, threshold=60, mask_scale=0.95): 
    """
    Finds the angle of the fly given an image.  Fly should be dark against a light background.

    Parameters:

      threshold   = value of threshold used to separate fly's from the (light) image background.

      mask_scale  = scales the radius of the circular mask around the fly's centroid. The value 
                    Should be in range [0,1]. 1 means the radius = min(image_width, image_height) 

    Returns: tuple (angle, angle_data) where

      angle      = the estimate of the fly's angle in radians
(https://github.com/willdickson/find_fly_angle
      angle_data = dictionary of useful data and images calculated during the angle estimation

      angle_data = { 
          'flipped': boolean which indicates whether or not fly's orientation was flipped,
          'moments': the moments of the maximum area contour in the thresholded image,
          'max_contour': the maximum area contour,
          'max_contour_area':  the area of the maximum area contour,
          'body_vector': unit vector along the fly's body axis,
          'contour_image': image with body contour, centroid, and fly's body axis drawn on it,
          'shifted_image': image shifted so that the fly's cendroid is at the center,
          'rotated_image': image rotated by the fly's body angle an shift so centroid is centered,
          'threshold_image': the thresholded image,
          'shifted_threshold_image': thresholded image shifted so the centroid is centered,
          'rotated_threshold_image': thresholded image rotated and shifted 
      }

    """

    # Get basic image data
    height, width = image.shape
    print(image.shape)
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height

    # Threshold, find contours and get contour with the maximum area
    # rval, threshold_image = cv2.threshold(image, threshold, np.iinfo(image.dtype).max, cv2.THRESH_BINARY_INV)
    # rval, threshold_image = cv2.threshold(image,0,256,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    rval, threshold_image = cv2.threshold(image, 25, np.iinfo(image.dtype).max, cv2.THRESH_BINARY_INV)
    contour_list, dummy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour, max_area = get_max_area_contour(contour_list)


    # Get moments and then compute centroid and rotation angle
    moments = cv2.moments(max_contour)
    centroid_x, centroid_y = get_centroid(moments)
    angle, body_vector = get_angle_and_body_vector(moments)

    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(centroid_x + axis_length*body_vector[0]), int(centroid_y + axis_length*body_vector[1])
    body_axis_pt_1 = int(centroid_x - axis_length*body_vector[0]), int(centroid_y - axis_length*body_vector[1])

    # Compute cirlce mask
    mask_radius = int(mask_scale*height/2.0)
    print(mask_radius)
    vals_x = np.arange(0.0,width)
    vals_y = np.arange(0.0,height)
    grid_x, grid_y = np.meshgrid(vals_x, vals_y)
    
    circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2

    # Draw image with body contours, centroid circle and body axis
    centroid = int(centroid_x), int(centroid_y)
    contour_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image,[max_contour],-1,(0,0,255),2)
    cv2.circle(contour_image, centroid, 10, (0,0,255), -1)
    cv2.line(contour_image, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)
    cv2.circle(contour_image, centroid, mask_radius, (0,0,255),2)

    # fly head indicator - does not indicate head or flip
    # cv2.circle(contour_image, body_axis_pt_0, 10, (0,255,255), -1) # to see fly's angle easily 05/18/2023

    # Get matrices for shifting (centering) and rotating the image
    shift_mat = np.matrix([[1.0, 0.0, (mid_x - centroid_x)], [0.0, 1.0, (mid_y - centroid_y)]]) 
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)

    # Shift and rotate the original image
    shifted_image = cv2.warpAffine(image, shift_mat, image_cvsize)
    rotated_image = cv2.warpAffine(shifted_image,rot_mat,image_cvsize)

    # Shift and rotate threshold image. 
    shifted_threshold_image = cv2.warpAffine(threshold_image, shift_mat, image_cvsize)
    rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
    rotated_threshold_image = rotated_threshold_image*circ_mask
    rval, rotated_threshold_image = cv2.threshold(rotated_threshold_image, threshold, np.iinfo(image.dtype).max, cv2.THRESH_BINARY)

    # Get orientation discriminant and flip image if needed 
    orient_ok, orient_discrim = is_orientation_ok(rotated_threshold_image,2)
    if not orient_ok:
        rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
        rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
        rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
        angle += np.deg2rad(-180.0)

    angle = normalize_angle_range(angle)

    #cv2.putText(contour_image,str(angle),org=centroid)

    # indicator of fly head (2/2/2024)
    cv2.circle(contour_image, (int(centroid[0] + np.cos(angle)*mask_radius),int(centroid[1] + np.sin(angle)*mask_radius)), 10, (0,255,255), -1)
    data = {
            'flipped': not orient_ok,
            'moments': moments,
            'max_contour': max_contour,
            'max_contour_area':  max_area,
            'body_vector': body_vector,
            'contour_image': contour_image,
            'shifted_image': shifted_image,
            'rotated_image': rotated_image,
            'threshold_image': threshold_image,
            'shifted_threshold_image': shifted_threshold_image,
            'rotated_threshold_image': rotated_threshold_image,
            }

    return angle, data 
        

def is_orientation_ok(image,k=2,is_first=True): 
    """
    Returns True if orienation is OK and False if fly need to be flipped. 

    Takes thresholded, shifted and rotated image and computes an orientation
    discriminant which is is the ratio of the k-th moments of area for each
    half of the body.  The discriminant should be such that it is > 1 for one
    orientation and < 1 for the other. 

    """

    mid_x, mid_y = int(0.5*image.shape[1]), int(0.5*image.shape[0])

    # Get moment for first body half 
    image_0 = np.array(image)
    image_0[:,:int(mid_x)] = 0
    image_0 = image_0[:,int(mid_x):]
    moment_0 = get_moment(image_0,k)

    # Get moment for second body half
    image_1 = np.array(image)
    image_1[:,int(mid_x):] = 0
    image_1 = np.fliplr(image_1)
    image_1 = image_1[:,int(mid_x):]
    moment_1 = get_moment(image_1,k)

    # Compute descriminant and flip flag
    discrim = (moment_0 - moment_1)/(moment_0 + moment_1)
    if discrim < 0:
        ok = False
    else:
        ok = True 
    return ok, discrim 



def get_moment(image,k=2):
    data = image.sum(axis=0)
    weighted_data = data*np.arange(data.shape[0])**k
    return  weighted_data.sum()


def get_max_area_contour(contour_list):
    """
    Given a list of contours finds the contour with the maximum area and 
    returns 
    """
    contour_areas = np.array([cv2.contourArea(c) for c in contour_list])
    max_area = contour_areas.max()
    max_ind = contour_areas.argmax()
    max_contour = contour_list[max_ind]
    return max_contour, max_area


def get_centroid(moments): 
    """
    Computer centroid given the image/blob moments
    """
    if moments['m00'] > 0:
        centroid_x = moments['m10']/moments['m00']
        centroid_y = moments['m01']/moments['m00']
    else:
        centroid_x = 0.0
        centroid_y = 0.0
    return centroid_x, centroid_y


def get_angle_and_body_vector(moments): 
    """
    Computre the angle and body vector given the image/blob moments
    """
    body_cov = np.array( [ [moments['mu20'], moments['mu11']], [moments['mu11'], moments['mu02'] ]])
    eig_vals, eig_vecs = np.linalg.eigh(body_cov)
    max_eig_ind = np.argmax(eig_vals**2)
    max_eig_vec = eig_vecs[:,max_eig_ind]
    angle = np.arctan2(max_eig_vec[1], max_eig_vec[0])
    # print("Angle 1:", angle, "Angle 2:",angle2, eig_vals[0]-eig_vecs[1][0], max(eig_vecs[1]))
    print("Angle:",np.rad2deg(angle),np.rad2deg(normalize_angle_range(angle)))
    print("Est Rotated Angle:", deg360to180(np.rad2deg(angle)+180),normalize_angle_range(normalize_angle_range(angle+np.pi)))
    return angle, max_eig_vec


def normalize_angle_range(angle):
    """
    Normalize the angle value so that is between -pi and pi
    """
    angle_adj = angle
    angle_adj += np.deg2rad(180.0)
    angle_adj = angle_adj % np.deg2rad(360.)
    angle_adj -= np.deg2rad(180.0)
    return angle_adj
def deg360to180(angle):
    # if angle <0:
    #     angle %=360
    # elif angle > 180:
    #     angle = 360 - angle
    if angle in range(-180,181):
        return angle
    else:
        angle%=360
        if angle>180:
            angle-=360
        return angle







