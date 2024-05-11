from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_1 = Image.open(r"C:\Users\ARNAV\Desktop\8.jpg")
im_2 = Image.open(r"C:\Users\ARNAV\Desktop\3.jpg")
im_1 = im_1.convert('1')
im_2 = im_2.convert('1')

new_size = (400, 400)
im_1_resized = im_1.resize(new_size)
im_2_resized = im_2.resize(new_size)

ar1 = np.array(im_1_resized)
ar2 = np.array(im_2_resized)
##print(ar1)
##print('**************************')
##print(ar2)

m1, n1 = ar1.shape[:2]
m2, n2 = ar2.shape[:2]

print(m1, n1, m2, n2)
size = m1 * n1


flag=0
count=0
if(m1==m2 and n1==n2):
 
 for i in range(0,m1-1): 
    for j in range(0,n1-1):
        if ar1[i][j]!=ar2[i][j]:
            flag=1
            count=count+1

 if(flag==1):
    print('not equal')
 else:
   print('equal')
 if(count!=0):
   #print(count)
   print("percentage of change",round(size/count,4))

else:
    print('not equal')


plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(im_1_resized, cmap='gray')
plt.title('Image 1')

plt.subplot(1, 2, 2)
plt.imshow(im_2_resized, cmap='gray')
plt.title('Image 2')

plt.show()

# Plot the comparison result
plt.figure()

if np.array_equal(ar1, ar2):
    plt.imshow(ar1, cmap='gray')
    plt.title('Images are equal')
else:
    plt.imshow(np.abs(ar1 ^ ar2), cmap='gray')
    plt.title('Images are not equal')

plt.show()    


