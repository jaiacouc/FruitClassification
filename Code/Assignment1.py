#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join


# In[8]:


# Read Images (e.g, .jpg, .png, and .tif)
peach_color = cv2.imread("C:/Users/itali/Assignment1_410/FIDS30/peaches/0.jpg")
plum_color = cv2.imread("C:/Users/itali/Assignment1_410/FIDS30/plums/7.jpg")
mango_color = cv2.imread("C:/Users/itali/Assignment1_410/FIDS30/mangos/2.jpg")

# Goes to directory and extracts the files out
path = 'C:/Users/itali/Assignment1_410/FIDS30/peaches'
files = [f for f in listdir(path) if isfile(join(path, f))]
# creates a image array based on the number of images in the directory
images_color = np.empty(len(files), dtype=object)
# adds images to the array
for n in range(0, len(files)):
    images_color[n] = cv2.imread(join(path, files[n]))


# In[9]:


# Cycles through the images and displays their channels
for n in range(0, len(images_color)):
    plt.imshow(images_color[n][:,:,0])
    plt.imshow(images_color[n][:,:,1])
    plt.imshow(images_color[n][:,:,2])

# Peaches color channels
plt.imshow(peach_color[:,:,0])
plt.imshow(peach_color[:,:,1])
plt.imshow(peach_color[:,:,2])

# Plum color channels
plt.imshow(plum_color[:,:,0])
plt.imshow(plum_color[:,:,1])
plt.imshow(plum_color[:,:,2])

# Mangos color channels
plt.imshow(mango_color[:,:,0])
plt.imshow(mango_color[:,:,1])
plt.imshow(mango_color[:,:,2])


# In[10]:


# Converts images to grey scale
images_gray = np.empty(len(images_color), dtype=object)
height =  np.empty(len(images_color), dtype=object)
width =  np.empty(len(images_color), dtype=object)
for n in range(0, len(images_color)):
    images_gray[n] = cv2.cvtColor(images_color[n], cv2.COLOR_BGR2GRAY)
    height[n], width[n] = images_gray[n].shape
    print(images_gray[n].shape)

# Convert images to grayscale.
peachG = cv2.cvtColor(peach_color, cv2.COLOR_BGR2GRAY)
heightPeG, widthPeG = peachG.shape
print(peachG.shape)

plumG = cv2.cvtColor(plum_color, cv2.COLOR_BGR2GRAY)
heightPlG, widthPlG = plumG.shape
print(plumG.shape)

mangoG = cv2.cvtColor(mango_color, cv2.COLOR_BGR2GRAY)
heightmG, widthmG = mangoG.shape
print(mangoG.shape)


# In[11]:


# Resize calculation by leveraging aspect ratio
images = np.empty(len(images_color), dtype=object)
for n in range(0, len(images_color)):
    tmp_height = height[n]
    tmp_width = width[n]
    aspect_ratio = tmp_width / tmp_height
    new_width = int(256 * aspect_ratio)
    while((new_width % 8) != 0):
        new_width = new_width + 1
    images[n] = cv2.resize(images_gray[n], dsize=(new_width, 256), interpolation=cv2.INTER_CUBIC)*255
    height[n], width[n] = images[n].shape
    print(images[n].shape)
    

# Raw Data Resizing
peach = cv2.resize(peachG, dsize=(320, 256), interpolation=cv2.INTER_CUBIC)
plum = cv2.resize(plumG, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
mango = cv2.resize(mangoG, dsize=(344, 256), interpolation=cv2.INTER_CUBIC)


peach = cv2.normalize(peach.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
plum = cv2.normalize(plum.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
mango = cv2.normalize(mango.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightPe, widthPe = peach.shape
heightPl, widthPl = plum.shape
heightM, widthM = mango.shape

print(peach.shape)
print(plum.shape)
print(mango.shape)


# In[12]:


# Plots images
for n in range(0, len(images_color)):
    plt.imshow(images[n], cmap=plt.get_cmap('gray'))
    plt.axis('off')

# Plot Images
plt.imshow(peach, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(plum, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(mango, cmap=plt.get_cmap('gray'))
plt.axis('off')


# In[13]:


# Saves images as a png
for n in range(0, len(images_color)):
    cv2.imwrite('C:/Users/itali/Assignment1_410/fruit_set/image' + str(n) + '.png', images[n])

# Save Images as PNG
cv2.imwrite('C:/Users/itali/Assignment1_410/Fruit/peach_grayTD.png', peach)
cv2.imwrite('C:/Users/itali/Assignment1_410/Fruit/plum_grayTD.png', plum)
cv2.imwrite('C:/Users/itali/Assignment1_410/Fruit/mango_grayTD.png', mango)


# In[14]:


# Some statistical information
print("Statistics: ", "Minimum:" ,peach.min(), "Maximum:" ,peach.max(), "Mean:" ,round(peach.mean(),3), "Std:" ,round(peach.std(),4))
print("Statistics: ", "Minimum:" ,plum.min(), "Maximum:" ,plum.max(), "Mean:" ,round(plum.mean(),3), "Std:" ,round(plum.std(),4))
print("Statistics: ", "Minimum:" ,mango.min(), "Maximum:" ,mango.max(), "Mean:" ,round(mango.mean(),3), "Std:" ,round(mango.std(),4))


# In[15]:


# Binarizes all images within the given folder
tmp_image = np.zeros(len(images_color), dtype=object)
threshold = np.zeros(len(images_color), dtype=object)
for n in range(len(images_color)):
    tmp_image[n] = np.zeros((height[n], width[n]), np.uint8)
    threshold[n] = images[n].mean()
    for i in range(height[n]):
        for j in range(width[n]):
            if(images[n][i][j]<threshold[n]):
                tmp_image[n][i][j] = 0
            else:
                tmp_image[n][i][j] = 255
    plt.imshow(tmp_image[n], cmap=plt.get_cmap('gray'))
    print(tmp_image[n])

# Binarize the image using a threshold
# Peach

tmpPe = np.zeros((heightPe, widthPe), np.uint8)
th1 = peach.mean()
for i in range(heightPe):
    for j in range(widthPe):
        if(peach[i][j]<th1):
            tmpPe[i][j] = 0
        else:
            tmpPe[i][j] = 255
plt.imshow(tmpPe, cmap=plt.get_cmap('gray'))
print(tmpPe)

# Plum
tmpPl = np.zeros((heightPl, widthPl), np.uint8)
th2 = plum.mean()
for i in range(heightPl):
    for j in range(widthPl):
        if(plum[i][j]<th2):
            tmpPl[i][j] = 0
        else:
            tmpPl[i][j] = 255
plt.imshow(tmpPl, cmap=plt.get_cmap('gray'))

# Mango
tmpM = np.zeros((heightM, widthM), np.uint8)
th3 = mango.mean()
for i in range(heightM):
    for j in range(widthM):
        if(mango[i][j]<th3):
            tmpM[i][j] = 0
        else:
            tmpM[i][j] = 255
plt.imshow(tmpM, cmap=plt.get_cmap('gray'))


# In[16]:


# Overlapping 8x8 for all images within the given folder.
im = np.zeros(len(images_color), dtype=object)
flat_im = np.zeros(len(images_color), dtype=object)
fspace = np.zeros(len(images_color), dtype=object)

for n in range(len(images_color)):
    im[n] = round(((height[n]-7)*(width[n]-7)))
    flat_im[n] = np.zeros((im[n], 65), np.uint8)
    k = 0
    for i in range(height[n]-7):
        for j in range(width[n]-7):
            crop = images[n][i:i+8, j:j+8]
            flat_im[n][k,0:64] = crop.flatten()
            k = k + 1
            
    fspace[n] = pd.DataFrame(flat_im[n])
    #fspace[n].to_csv('C:/Users/itali/Assignment1_410/DataFrames/fspace' + str(n) + '.csv', index=False)
            

# Overlaping 8x8
# Peach
pe = round(((heightPe-7)*(widthPe-7)))
flatPe = np.zeros((pe, 65), np.uint8)
k = 0
for i in range(heightPe-7):
    for j in range(widthPe - 7):
        crop_tmp1 = peach[i:i+8, j:j+8]
        flatPe[k,0:64] = crop_tmp1.flatten()
        k = k + 1

# Creates Panda object
fspacePe = pd.DataFrame(flatPe)
fspacePe.to_csv('C:/Users/itali/Assignment1_410/DataFrames/fspacePe.csv', index=False)

# Plum
pl = round(((heightPl-7)*(widthPl-7)))
flatPl = np.ones((pl, 65), np.uint8)
k = 0
for i in range(heightPl-7):
    for j in range(widthPl - 7):
        crop_tmp = plum[i:i+8, j:j+8]
        flatPl[k,0:64] = crop_tmp.flatten()
        k = k + 1

# Creates Panda object
fspacePl = pd.DataFrame(flatPl)
fspacePl.to_csv('C:/Users/itali/Assignment1_410/DataFrames/fspacePl.csv', index=False)

# Mango
m = round(((heightM-7)*(widthM-7)))
flatM = (np.ones((m, 65), np.uint8)+1)
k = 0
for i in range(heightM-7):
    for j in range(widthM - 7):
        crop_tmp2 = mango[i:i+8, j:j+8]
        flatM[k,0:64] = crop_tmp2.flatten()
        k = k + 1

# Creates Panda object
fspaceM = pd.DataFrame(flatM)
fspaceM.to_csv('C:/Users/itali/Assignment1_410/DataFrames/fspaceM.csv', index=False)


# In[17]:


# Non-Overlaping 8x8
# Peach
pe = round((heightPe)*(widthPe)/64)
flat_pe = np.zeros((pe, 65), np.uint8)
k = 0
for i in range(0,heightPe,8):
    for j in range(0,widthPe,8):
        crop_tmp3 = peach[i:i+8, j:j+8]
        flat_pe[k,0:64] = crop_tmp3.flatten()
        k = k + 1

# Creates Panda obj
fspace_pe = pd.DataFrame(flat_pe)
fspace_pe.to_csv('C:/Users/itali/Assignment1_410/DataFrames/fspace_Pe.csv', index=False)

# Plum
pl = round((heightPl)*(widthPl)/64)
flat_pl = np.ones((pl, 65), np.uint8)
k = 0
for i in range(0,heightPl,8):
    for j in range(0,widthPl,8):
        crop_tmp4 = plum[i:i+8, j:j+8]
        flat_pl[k,0:64] = crop_tmp4.flatten()
        k = k + 1

# Creates Panda obj
fspace_pl = pd.DataFrame(flat_pl)
fspace_pl.to_csv('C:/Users/itali/Assignment1_410/DataFrames/fspace_Pl.csv', index=False)

# Mango
m = round((heightM)*(widthM)/64)
flat_m = (np.ones((m, 65), np.uint8)+1)
k = 0
for i in range(0,heightM,8):
    for j in range(0,widthM,8):
        crop_tmp5 = mango[i:i+8, j:j+8]
        flat_m[k,0:64] = crop_tmp5.flatten()
        k = k + 1

# Creates Panda obj
fspace_m = pd.DataFrame(flat_m)
fspace_m.to_csv('C:/Users/itali/Assignment1_410/DataFrames/fspace_m.csv', index=False)


# In[18]:


# Statisitcal Info and Graphs
# Mean of the features
mango_mean = fspace_m.iloc[:,0:63].mean()
peach_mean = fspace_pe.iloc[:,0:63].mean()
plum_mean = fspace_pl.iloc[:,0:63].mean()

fig = plt.figure(figsize = (10, 5))
plt.plot(mango_mean, label = 'Mango', color = 'red')
plt.plot(plum_mean, label = 'Plum', color = 'green')
plt.plot(peach_mean, label = 'Peach')
plt.xlabel('Features')
plt.ylabel('Mean')
plt.title('Means of Mangos, Peaches, and Plums')
plt.grid(True)
plt.legend()
plt.show()

# Number of observations
data = {'Mango':len(fspace_m), 'Plum':len(fspace_pl), 'Peach':len(fspace_pe)}
fruit = list(data.keys())
obs = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(fruit, obs, color ='b',
        width = 0.4)
 
plt.xlabel("Fruits")
plt.ylabel("Number of Observations")
plt.title("Observations per fruit")
plt.show()

#2D plot of features 1 and 54
fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(111)
ax1.scatter(fspace_pl.loc[:,1], fspace_pl.loc[:,54], color = 'green', s = 1, label= 'Plum')
ax1.scatter(fspace_pe.loc[:,1], fspace_pe.loc[:,54], s= 1, label= 'Peach')
#ax1.scatter(fspaceM.loc[:,1], fspaceM.loc[:,54], color = 'red', s=1)
plt.xlim(10, 250)
plt.ylim(10, 250)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 54')
ax1.legend()
plt.show()

# 3D plot of features 1, 22, and 54
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot(1,1,1, projection= '3d')
ax.scatter(fspace_pl.loc[:,1], fspace_pl.loc[:,22], fspace_pl.loc[:,54], color = 'green', s=1, label = 'Plum')
ax.scatter(fspace_pe.loc[:,1], fspace_pe.loc[:,22], fspace_pe.loc[:,54], s=1, label = 'Peach')
ax.scatter(fspace_m.loc[:,1], fspace_m.loc[:,22], fspace_m.loc[:,54], color = 'red', s=1, label = 'Mango')
ax.set_xlabel('Feature 22')
ax.set_ylabel('Feature 1')
ax.set_zlabel('Feature 54')
ax.legend()
plt.show()


# In[19]:


# Merges all data frames and converts to a csv
mged_set = pd.concat(fspace)
mged_set.to_csv('C:/Users/itali/Assignment1_410/DataFrames/merged_set.csv')

# Merge features from classes
frames01 = [fspacePe, fspacePl]
mged01 = pd.concat(frames01)

indx01 = np.arange(len(mged01))
rndmged01 = np.random.permutation(indx01)

rndmged01=mged01.sample(frac=1).reset_index(drop=True)

rndmged01.to_csv('C:/Users/itali/Assignment1_410/DataFrames/merged01.csv')


# In[20]:


# Creates a csv from the 3 dataframes.
frames012 = [fspacePe, fspacePl, fspaceM]
mged012 = pd.concat(frames012)
indx012 = np.arange(len(mged012))
rndmged012 = np.random.permutation(indx012)

rndmged012=mged01.sample(frac=1).reset_index(drop=True)
mged012.to_csv('C:/Users/itali/Assignment1_410/DataFrames/merged012.csv')


# In[ ]:




