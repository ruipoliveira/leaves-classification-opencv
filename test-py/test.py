import cv2
import numpy as np




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
print get_corner_points(img,100)
cv2.imshow('image',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

