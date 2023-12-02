# -*- coding: utf-8 -*-
import sys
sys.path.append("../src/")

import time
import rasterio
import numpy as np
import starfm4py as stp
import matplotlib.pyplot as plt
from PIL import Image
from parameters import (path, sizeSlices, resize_params)



start = time.time()

# Assign path values
lsat0_path = 'Test_9\lsatRGB-2023-01-12-SR_B7.tif'
modis0_path = 'Test_9\modisRGB-2023-01-12-sur_refl_b07.tif'
modis1_path = 'Test_9\modisRGB-2023-02-13-sur_refl_b07.tif'

# Resize the tif images
lsat0 = Image.open(lsat0_path).resize(resize_params)
modis0 = Image.open(modis0_path).resize(resize_params)
modis1 = Image.open(modis1_path).resize(resize_params)
lsat0.save(lsat0_path)
modis0.save(modis0_path)
modis1.save(modis1_path)

#Set the path where the images are stored
product = rasterio.open(lsat0_path)
profile = product.profile
LandsatT0 = rasterio.open(lsat0_path).read(1)
MODISt0 = rasterio.open(modis0_path).read(1)
MODISt1 = rasterio.open(modis1_path).read(1)

# Set the path where to store the temporary results
path_fineRes_t0 = 'Temporary/Tiles_fineRes_t0/'
path_coarseRes_t0 = 'Temporary/Tiles_coarseRes_t0/'
path_coarseRes_t1 = 'Temporary/Tiles_fcoarseRes_t1/'

# Flatten and store the moving window patches
fine_image_t0_par = stp.partition(LandsatT0, path_fineRes_t0)
coarse_image_t0_par = stp.partition(MODISt0, path_coarseRes_t0)
coarse_image_t1_par = stp.partition(MODISt1, path_coarseRes_t1)

print ("Done partitioning!")

# Stack the the moving window patches as dask arrays
S2_t0 = stp.da_stack(path_fineRes_t0, LandsatT0.shape)
S3_t0 = stp.da_stack(path_coarseRes_t0, MODISt0.shape)
S3_t1 = stp.da_stack(path_coarseRes_t1, MODISt1.shape)

shape = (sizeSlices, LandsatT0.shape[1])

print ("Done stacking!")

# Perform the prediction with STARFM
for i in range(0, LandsatT0.size-sizeSlices*shape[1]+1, sizeSlices*shape[1]):
    
    fine_image_t0 = S2_t0[i:i+sizeSlices*shape[1],]
    coarse_image_t0 = S3_t0[i:i+sizeSlices*shape[1],]
    coarse_image_t1 = S3_t1[i:i+sizeSlices*shape[1],]
    prediction = stp.starfm(fine_image_t0, coarse_image_t0, coarse_image_t1, profile, shape)
    
    if i == 0:
        predictions = prediction
        
    else:
        predictions = np.append(predictions, prediction, axis=0)
  

# Write the results to a .tif file   
print ('Writing product...')
profile = product.profile
profile.update(dtype='float64', count=1) # number of bands
current_test = lsat0_path[:lsat0_path.index('\\')]
file_name = path + current_test + '-prediction.tif'

result = rasterio.open(file_name, 'w', **profile)
result.write(predictions, 1)
result.close()

pred_img = Image.fromarray(predictions)
if (pred_img.mode != 'RGB'): 
    pred_img = pred_img.convert('RGB')
pred_img.save(path + current_test + '-prediction.jpeg')


end = time.time()
print ("Done in", (end - start)/60.0, "minutes!")

# Display input and output
plt.imshow(LandsatT0)
plt.gray()
plt.show()
plt.imshow(MODISt0)
plt.gray()
plt.show()
plt.imshow(MODISt1)
plt.gray()
plt.show()	
plt.imshow(predictions)
plt.gray()
plt.show()
