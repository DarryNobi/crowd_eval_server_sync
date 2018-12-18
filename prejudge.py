import cv2
import matplotlib.pyplot as plt
import numpy as np
# path='test.jpg'
path='blue.png'
img = cv2.imread(path)
img=cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('',img_gray)
#hist=cv2.calcHist(img_gray, [0], None, [64], [0,256])
hist,_=np.histogram(img_gray.ravel(),32,[0,256])
max_hist=np.max(hist)
print(max_hist)
plt.plot(hist)
plt.show()