#!/usr/bin/env python3
import numpy as np 
import cv2
from matplotlib import pyplot as plt



img = cv2.imread('AR.jpg',0)

dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
shift = np.fft.fftshift(dft)

magnitude_spectrum = 100 * np.log(cv2.magnitude(shift[:, :, 0], shift[:, :, 1]))

rows, cols = np.shape(img)
row, column = int(rows / 2), int(cols / 2)  # center



mask = np.ones((rows, cols, 2), np.uint8)
r = 80
center = [row, column]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0

# apply mask and inverse DFT
fshift = shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_ifft = cv2.idft(f_ishift)
img_ifft = cv2.magnitude(img_ifft[:, :, 0], img_ifft[:, :, 1])




fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(2,1,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,1,2)
ax2.imshow(img_ifft, cmap='gray')
ax2.title.set_text('Tag detected')
plt.show()



'''
img = cv2.imread('AR.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
print(corners)
'''
# Now draw them
'''
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]
cv2.imshow('subpixel5.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
