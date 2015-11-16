import numpy as np
import cv2
from matplotlib import pyplot as plt

Xsize, Ysize = 400, 300
             #(template, searchableImg)
def findObject(img1,img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    MIN_MATCH_COUNT = 10


    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w,temp = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        pts1 = np.float32(dst)
        pts2 = np.float32([[0,0],[0,Ysize],[Xsize,Ysize],[Xsize,0]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        foundedObject = cv2.warpPerspective(img2,M,(Xsize,Ysize))

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None


    return foundedObject


def main():
    files = 6
    
    template = './template.jpg'
    img1 = cv2.imread(template)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    for i in range(files):
        path = './images1/' + str(i+1) + '.jpg'
        savePath = './TESTS/sift/images1-flann/SIFT-test'
               
        img2 = cv2.imread(path)  
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
        img3 = findObject(img1, img2)
        plt.imshow(img3)
            
        path = savePath + str(i+1) + '.jpg'
        plt.imsave(path, img3)
        print 'progress: ' + str((i+1)/(files*1.0) * 100) + '%'

main()