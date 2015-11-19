import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pytesseract
from PIL import Image


Xsize, Ysize = 510*2, 300*2
nameX1, nameX2 = 260, 300
nameY1, nameY2 = 375, 780
subNameX1, subNameX2 = 185, 230
subNameY1, subNameY2 = nameY1, nameY2
faceX1, faceX2 = 165, 500
faceY1, faceY2 = 70, 360
signX1, signX2 = 440, 535
signY1, signY2 = 380, 750

files = 16
template = './template.jpg'
#zdjecia powinny byc w notacji i.jpg dla i = 1, 2, 3...
dirPath = './images/'           #input
saveDirPath = './TESTS/' + dirPath  #output

def getName(img):
    return img[nameX1:nameX2, nameY1:nameY2]

def getSubName(img):
    return img[subNameX1:subNameX2, subNameY1:subNameY2]

def getFace(img): #xD
    return img[faceX1:faceX2, faceY1:faceY2]

def getSign(img):
    return img[signX1:signX2, signY1:signY2]

def img2str(img):
    cv2.imwrite('temp.jpg', img)
    return pytesseract.image_to_string(Image.open('temp.jpg'),lang='pol')

def transform2default(img, dst):
    pts1 = np.float32(dst)
    pts2 = np.float32([[0,0],[0,Ysize],[Xsize,Ysize],[Xsize,0]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(Xsize,Ysize))

def connectVerticaly(imgUP, imgDOWN):
    h1, w1 = imgUP.shape[:2]
    h2, w2 = imgDOWN.shape[:2]
    if type(imgUP[0][0]) != type(np.array(0)):
        vis = np.zeros(((h1+h2), max(w1,w2)), np.uint8)
    else:
        vis = np.zeros(((h1+h2), max(w1,w2), 3), np.uint8)
    vis[:h1, :w1] = imgUP
    vis[h1:h2+h1, :w2] = imgDOWN
    return vis

def binarization(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) != 2:
        print 'wrong input pic binarization()'
        return None

    h, w = img.shape
    white, black = 255, 0


    # ret3,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    for i in range(0, h):
        for j in range(0, w):
            if img.item(i,j) <= 25: #na podstawie histogramu powinien znajdowac ta wartosc
                val = black
            else:
                val = white
            img.itemset((i,j),val) # img[i][j] = val #
    kernel = np.ones((4,4),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return img






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

    print 'flann.knnmatch starting!'
    matches = flann.knnMatch(des1,des2,k=2) # czasem wyjebuje 139 (sigsegv) i chuj wie czemu :)
    print 'flann.knnmatch done'

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

        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        foundedObject = transform2default(img2, dst)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    return foundedObject


def main():
    # for i in range(files):
    #     path = dirPath + str(i+1) + '.jpg'
    #
    #     img2 = cv2.imread(path)
    #     img3 = findObject(img1, img2)
    #     plt.imshow(img3)
    #
    #     path = saveDirPath + str(i+1) + '.jpg'
    #     cv2.imwrite(path, img3)
    #     print 'progress: ' + str((i+1)/(files*1.0) * 100) + '%' + '  file: ' + str(i+1)

    fileName = '13.jpg'
    dirPath = './TESTS/images/'
    img1 = cv2.imread(template)

    path = dirPath + fileName
    img2 = cv2.imread(path)
    cv2.imwrite('./subname.jpg', getSubName(img2))

    img1 = cv2.imread(template)
    if not os.path.exists(saveDirPath):
        os.makedirs(saveDirPath)
    if not os.path.exists(dirPath):
        print 'input directory not exist'
        return 1

    if len(img2.shape) == 3:
        for i in range(0, img2.shape[0]):
            for j in range(0, img2.shape[1]):
                img2.itemset((i,j,1),255)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img2 = cv2.medianBlur(img2,3)
    img2 = cv2.equalizeHist(img2)

    hist,bins = np.histogram(img2.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img2
             .flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

    img2 = binarization(img2)
    img3 = img2
    # img3 = findObject(img1, img2)
    #
    name = binarization(getName(img3))
    subName = binarization(getSubName(img3))
    #
    imgUp = connectVerticaly(getFace(img3), binarization(getSign(img3)))
    imgDown = connectVerticaly(name, subName)
    #
    print img2str(name).upper()
    print img2str(subName).upper()
    cv2.imshow('ID', connectVerticaly(imgUp,imgDown))

    cv2.imshow('xD', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
print 'done'