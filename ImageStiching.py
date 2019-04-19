from scipy.ndimage import filters
import numpy as np
import cv2
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def HarrisCornerDetector(img):
	sd = 1 # Standard deviation
	#Ix = filters.gaussian_filter(img, sd, (0, 1))
	#Iy = filters.gaussian_filter(img, sd, (1, 0))
	kernel = np.array([[-1,0,1]])
	Ix = cv2.filter2D(img, -1, kernel)
	Iy = cv2.filter2D(img, -1, np.transpose(kernel))

	Sx2 = filters.gaussian_filter(Ix*Ix, sd)
	Sy2 = filters.gaussian_filter(Iy*Iy, sd)
	Sxy = filters.gaussian_filter(Ix*Iy, sd)

	det = Sx2*Sy2 - Sxy*Sxy
	trace = Sx2 + Sy2
	R = det - 0.04*(trace**2) # k = 0.04
	return R

def R_value(R_elem):
	return R_elem[2]


def NonmaxSuppresion(R):
	featureP = []
	for i in range(R.shape[0]):
		for j in range(R.shape[1]):
			featureP.append([i,j,R[i][j]])
	featureP.sort(key=R_value,reverse=True)
	
	numFTP,MaxFTP = 0,200
	checkArea,size = np.zeros(R.shape),30
	FTP = np.zeros((MaxFTP,2))
	for i in range(len(featureP)):
		p = featureP[i][0:2]
		if checkArea[p[0],p[1]] == 0:
			FTP[numFTP] = p
			numFTP += 1
			#Caanot pick another feature point near these (2*size+1)*(2*size+1) area
			xMin, xMax = max(0,p[0]-size), min(R.shape[0],p[0]+size)
			yMin, yMax = max(0,p[1]-size), min(R.shape[1],p[1]+size)
			checkArea[xMin:xMax, yMin:yMax] = 1

		if numFTP==MaxFTP:
			break
	FTP = FTP[0:numFTP,:] # reduce the shape from (MaxFTP,2) to (numFTP,2)
	return np.asarray(FTP)




if __name__ == "__main__":
	logging.basicConfig(
		format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
		level=logging.INFO, datefmt='%m-%d %H:%M:%S'
	)
	logging.info("Load Image...")
	imgName = 'park.jpg' #'1.png'
	ori_img = cv2.imread(imgName,cv2.IMREAD_GRAYSCALE)
	img = np.float32(ori_img)

	logging.info("Perform Harris Corner Detector")
	R = HarrisCornerDetector(img)

	logging.info("Nonmax Suppresion...")
	featureP = NonmaxSuppresion(R)

	logging.info("Show the iamge with Feature Points...")
	img=mpimg.imread(imgName)
	imgplot = plt.imshow(img)
	# The order of x and y are different with images order
	plt.scatter(featureP[:,1],featureP[:,0], marker='+',c='r')
	plt.show()

