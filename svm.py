from PIL import Image
from collections import Counter
import numpy as np
import cv2
import matplotlib.pyplot as plt

myimg1 = Image.open(r"C:\Users\ARNAV\Desktop\3.jpg")
array1 = np.asarray(myimg1)
print(np.shape(array1))

myimg2 = Image.open(r"C:\Users\ARNAV\Desktop\8.jpg")
array2 = np.asarray(myimg2)
print(np.shape(array2))


flat_array_1 = array1.flatten()
print(np.shape(flat_array_1))

flat_array_2 = array2.flatten()
print(np.shape(flat_array_2))

RH1 = Counter(flat_array_1)
H1 = []
for i in range(256):
    if i in RH1.keys():
        H1.append(RH1[i])
    else:
       H1.append(0)

RH2 = Counter(flat_array_2)
H2 = []
for i in range(256):
    if i in RH2.keys():
        H2.append(RH2[i])
    else:
       H2.append(0)
       
def L2Norm(H1, H2):
    distance = 0
    for i in range(len(H1)):
        distance += np.square(float(H1[i]) - float(H2[i]))
    return np.sqrt(distance)


dist_test_ref_1 = L2Norm(H1,H2)
print("The distance between Reference_Image_1 and Test Image is : {}".format(dist_test_ref_1))
dist_test_ref_2 = L2Norm(H2,H1)
print("The distance between Reference_Image_2 and Test Image is : {}".format(dist_test_ref_2))
myimg1 = cv2.imread(r'C:\Users\ARNAV\Desktop\3.jpg')
myimg2 = cv2.imread(r'C:\Users\ARNAV\Desktop\8.jpg')
hist_img1 = cv2.calcHist([array1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
hist_img1[255, 255, 255] = 0
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

hist_img2 = cv2.calcHist([array2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
hist_img2[255, 255, 255] = 0
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
print("Similarity Score: ", round(metric_val,2))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(array1)
plt.title("Reference Image 1")

plt.subplot(1, 2, 2)
plt.imshow(array2)
plt.title("Test Image")

plt.show()

# Plotting histograms
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(hist_img1, color='blue')
plt.title("Histogram - Reference Image 1")

plt.subplot(1, 2, 2)
plt.plot(hist_img2, color='green')
plt.title("Histogram - Test Image")

plt.show()
