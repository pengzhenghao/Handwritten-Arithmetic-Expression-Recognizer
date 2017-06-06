# This program is used to preprocess the input image and output a [28,196] image.
import os
import re

import cv2
import numpy as np


class Preprocessor(object):
    def __init__(self):
        self.height = 100
        self.half_height = int(self.height / 2)

    def generateTrainData(self,img_file,save_dir):
        return self._do(img_file,save_dir)



    def _process(self, img_file=None):
        '''
        :param image:
        :return:
        '''
        if img_file == None:
            print('What are you doing!')
            return

        image = cv2.imread(img_file)
        # cv2.imshow('img',image)
        # cv2.waitKey(2000)
        print(image.shape)
        if image.shape[0]<image.shape[1]:
            image = cv2.resize(image, (1000,(int(image.shape[0] * 1000 / image.shape[1]))),
                               interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, (int(image.shape[1] * 1000. / image.shape[0]), 1000), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        cv2.imshow('img',gray)
        # cv2.waitKey(20000)
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        # blur and threshold the image
        blurred = cv2.blur(gradient, (4, 4))
        cv2.imshow('bliur',blurred)
        # cv2.waitKey(0)
        (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



        # perform a series of erosions and dilations
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=10)
        cv2.imshow('thr',thresh)
        # cv2.waitKey(20000)

        rect = self._getBoundingRect(thresh)
        # print(rect,'rrrr'/])

        box = np.int0(cv2.boxPoints(rect))






        # draw a bounding box arounded the detected barcode and display the image
        cv2.drawContours(image, [box], -1, (255, 255, 255), 3)
        cv2.imshow('image',image)
        cv2.waitKey(1000)


        # gray = cv2.equalizeHist(gray)
        # cv2.imshow('img',gray)
        # cv2.waitKey(20000)

        # cv2.imshow('gray',thresh)
        # cv2.waitKey(1000)
        cutoff = self._crop(gray, rect,True)
        # cv2.imshow('cutoff',cutoff)
        # cv2.waitKey(2000)
        _,cutoff = cv2.threshold(cutoff, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # cutoff = cv2.equalizeHist(cutoff)
        cv2.imshow('cutf',cutoff)
        # cv2.waitKey(2000)

        # cutoff = cv2.erode(cutoff, None, iterations=1)
        # cutoff = self._revert(cutoff)
        cutoff = self._padding(cutoff)
        return cutoff

    def _do(self, filename=None,save_dir='./tmp',dialate=0,erode=0):

        img = self._process(filename)
        cv2.imshow('before',img)
        # cv2.waitKey(2000)
        # cv2.waitKey(0)

        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 199, 10)

        # img = cv2.erode(img,None,1)
        # img = cv2.dilate(img,None,1)

        img = cv2.resize(img, (int(img.shape[1] * self.height / img.shape[0]), self.height),
                         interpolation=cv2.INTER_AREA)

        # img = cv2.erode(img,None,3)


        #
        # ler
        # # 闭运算
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


        # img = self._padding(img)

        # cv2.imshow('cutf',img)
        # cv2.waitKey(0)







        _, contours, __ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours_list = []

        for con in contours:
            # box = np.int0(cv2.boxPoints(cv2.minAreaRect(con)))

            # draw a bounding box arounded the detected barcode and display the image
            # cv2.drawContours(img, [box], -1, (255, 255, 255), 3)


            M = cv2.moments(con)
            contours_list.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), con, M["m00"]])
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        contours_list = sorted(contours_list, key=lambda x: x[0])

        print(len(contours_list))

        # dealing with each contour:
        for ind, contour in enumerate(contours_list):
            tmp = self._extract(img, contour)
            # tmp = self._crop(img,contour[2])
            tmp = self._resize_and_padding(tmp, (28, 28),ratio=1)
            cv2.imshow(str(ind), tmp)
            cv2.waitKey(1000)

            cv2.imwrite(save_dir+'/%d.bmp'% (self._getLastSaveFileIndex(save_dir)+1), tmp)

        return 'Success!'

    def _getLastSaveFileIndex(self, dir):
        result = []
        for _, _, files in os.walk(dir):
            if files != []:
                for name in files:
                    if re.search('.bmp',name)==None:
                        continue
                    result.append(int(name.split('.')[0]))

        if result == []:
            return 0
        # print(result.group()[0])
        return max(result)


    def _extract(self, img, packed_contour):
        mask = cv2.drawContours(np.zeros(img.shape), [packed_contour[2]], -1, 255, cv2.FILLED) * img
        # rect = ((packed_contour[0], self.half_height), (self.height, self.height), 0)
        rect = cv2.minAreaRect(packed_contour[2])
        return self._crop(mask, rect, False)

        # return self._crop(img,cv2.minAreaRect(packed_contour[2]))

    def _resize_and_padding(self, img, size=(28, 28), ratio=1):
        img = self._padding(img.copy(), ratio=ratio)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    def _padding(self, img, ratio=30):
        a, b = img.shape
        c, d = 0, 0
        if ratio * a > b:
            c = ratio * a - b
        else:
            d = b / ratio - a
        return cv2.copyMakeBorder(img, int(d / 2), int(d - int(d / 2)), int(c / 2), int(c - int(c / 2)),
                                  cv2.BORDER_CONSTANT, value=0)

    def _getBoundingRect(self, img):
        (_, cnts, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        result = []
        for contour in cnts:
            for j in contour:
                result.append(j[0])
        return cv2.minAreaRect(np.array(result))

    def _crop(self, img, rect, rotated=True):
        if rotated:


            if rect[2]<-45.:
                M = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), 90+rect[2], 1)
                rectg0 = tuple(np.dot(M, (rect[0][0], rect[0][1], 1))[0:2])
                rectg1 = tuple(np.dot(M, (rect[1][1], rect[1][0], 1))[0:2])
            else:
                M = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), rect[2], 1)
                rectg0 = tuple(np.dot(M, (rect[0][0], rect[0][1], 1))[0:2])
                rectg1 = tuple(np.dot(M, (rect[1][0], rect[1][1], 1))[0:2])
            dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            return dst[int(rectg0[1] - rectg1[1] / 2 ):int(rectg0[1] + rectg1[1] / 2 ),
                   int(rectg0[0] - rectg1[0] / 2 ):int(rectg0[0] + rectg1[0] / 2) ]
        else:
            dst = img.copy()
            rectg0 = rect[0]
            rectg1 = rect[1]
        return dst[int(rectg0[1] - rectg1[1] / 2 + 10):int(rectg0[1] + rectg1[1] / 2 - 10),
               int(rectg0[0] - rectg1[0] / 2 + 10):int(rectg0[0] + rectg1[0] / 2 - 10)]

    def _revert(self, img):
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                img[i, j] = 255 - img[i, j]
        return img





#
#
#
# ind = 0
# for _, _, files in os.walk('./'):
#     if files != []:
#         for name in files:
#             if re.search('.bmp', name) == None:
#                 continue
#             result.append(int(name.split('.')[0]))










a = Preprocessor()
a._do('./tmp/lbj.jpeg','./tmp')
#training_data/multi
cv2.destroyAllWindows()
# for root, _, files in os.walk('./tmp'):
#     for i in files:
#         print(root+i)
#         if re.search('.jpeg',i)==None:
#             continue
#         a._do(root+'/'+i,'./002')
# #
