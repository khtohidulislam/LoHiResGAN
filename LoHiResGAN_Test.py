#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kh Tohidul Islam; Monash Biomedical Imaging, Monash University, Victoria, Australia (KhTohidul.Islam@monash.edu)
"""

import nibabel as nib #5.0.0
import numpy as np #1.22.3
import tensorflow as tf #2.7.0
import cv2 # 4.6.0
import os
import glob

# Load Model
model = tf.keras.models.load_model('/.....LoHiResGAN/Trained_Model_T1/')

# Load the input NIfTI file
input_dir = "/.....LoHiResGAN/Test_Data_T1/"
output_dir = "/.....LoHiResGAN/Synt_Output/"
# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over each NIFTI file in the input directory
for file_path in glob.glob(os.path.join(input_dir, '*.nii.gz')):
    # Load the NIFTI file
    input_image = nib.load(file_path)
    input_data = input_image.get_fdata()
    
    normalized_arr = 2 * (input_data - input_data.min()) / (input_data.max() - input_data.min()) - 1
    
    
    # Create an empty array to store the processed slices
    output_data = np.zeros_like(input_data)
    X = input_data.shape[0]
    Y = input_data.shape[1]
    
    # Iterate through each slice, process it, and save it to the output array
    for i in range(normalized_arr.shape[2]):
        # Get the i-th slice
        slice_data = normalized_arr[:, :, i]
        img = cv2.rotate(slice_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
        resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    
        resized_slice_data = np.expand_dims(resized_img, -1)
        resized_slice_data = np.expand_dims(resized_slice_data, -0)
        
        gen_image = model.predict(resized_slice_data)
        
        gen_image = np.squeeze(gen_image, axis=0)
        
        
        gen_image = np.squeeze(gen_image, axis=-1)
        
        rescaled_arr = (gen_image + 1) * 127.5
        rescaled_arr = rescaled_arr.astype('float64')
       
        normalized_gen_image_float = np.rot90(rescaled_arr, k=-1)
        
        resized_slice_data_final = cv2.resize(normalized_gen_image_float, (Y, X), interpolation=cv2.INTER_CUBIC)
    
        output_data[:, :, i] = resized_slice_data_final
    
    # Create a new NIfTI image with the processed data and the same header as the input image
    output_image = nib.Nifti1Image(output_data, input_image.affine, header=input_image.header)
    
    # Save the output image to a new NIfTI file
    filename = os.path.splitext(os.path.basename(file_path))[0]  # Get the base file name
    output_filename = os.path.join(output_dir, filename + '.gz')  # Construct the full output file path with .nii.gz extension
    nib.save(output_image, output_filename)  # Use gzip compression with compression level 1
