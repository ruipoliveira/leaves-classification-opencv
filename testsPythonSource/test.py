import cv2
import numpy as np
import imutils

im = cv2.imread('/home/roliveira/Documents/leaves-classification/image_leaves/Viburnum.jpg') 


imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,1)



image,contours, hierarchy = cv2.findContours(thresh,1,2)

print len(image)
#contours=contours[1::]
cv2.drawContours(imgray,contours,-1,(200,200,0),3)

cv2.imshow('image',imgray)
cv2.waitKey(0)


'''

def get_cnt(img):
	ret,thresh = cv2.threshold(img,127,255,0)
	img,contours,hierarchy = cv2.findContours(thresh, 1, 2)

	cnt = contours[0]
	return cnt

def get_moments(img):
	return cv2.moments(get_cnt(img))

def get_solidity(img):
	print "IMAGEEEE! get_solidity "
	print img
	cnt = get_cnt(img)
	area = cv2.contourArea(cnt)
	hull = cv2.convexHull(cnt)
	hull_area = cv2.contourArea(hull)
	if hull_area == 0:
		hull_area = 0.0001
	solidity = float(area)/hull_area

	return solidity


img = cv2.imread('/home/roliveira/Documents/leaves-classification/image_leaves/Acer_platanoides.jpg',0)
ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


def get_corner_points(img, maxFeat):
	#print img
	#print maxFeat

	feature_params = dict( maxCorners = maxFeat, qualityLevel = 0.6, minDistance = 7, blockSize = 7 )
	corners = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)
	return corners


contours,hierarchy,dsa = cv2.findContours(thresh, 1, 2)
 
cnt = contours[0]
M = cv2.moments(cnt)
print M

print get_solidity(thresh)
print get_moments(img)
print get_corner_points(img,100)'''
'''
image = cv2.imread('/home/roliveira/Documents/leaves-classification/image_leaves/Acer_platanoides.jpg',0)

image = cv2.resize(image,(400,500))

gray = image.copy()

(thresh, im_bw) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY )

derp,contours,hierarchy = cv2.findContours(im_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

cnts = max(cnts, key=cv2.contourArea)

cv2.drawContours(image, [cnts], -1, (0, 255, 255), 2)


cv2.imshow('image',image)
cv2.waitKey(0)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(image, (5, 5), 0)

thresh = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

c = max(cnts, key=cv2.contourArea)

extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])


cv2.drawContours(image, [c], -1, (0, 255, 255), 2)

#ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#contours,hierarchy,dsa = cv2.findContours(thresh, 1, 2)

#print contours

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''