# 本程序提供一个Preprocessor类, 它可以将一张完整的相片裁剪成单个字符一张的小图。
import os
import re
import math
import cv2
import numpy as np



# 网上复制的代码, 用于将图像旋转90度, 方便手机拍摄上传图像。
def rotate_about_center(src, angle=90, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

class Preprocessor(object):

    def __init__(self):
        pass

    # 用于生成测试数据, 自动将拍摄图像裁剪成若干小图。
    def generateTrainData(self,img_file,save_dir='./tmp1'):
        return self._do(img_file,save_dir=save_dir)

    # 用于实时处理数据, 输入为一幅图像。
    def onlineTest(self,img):
        return self._do(img=img)


    def _process(self, img_file=None, img=None):

        image = cv2.imread(img_file) if img==None else img

        if type(image)=='NoneType':
            return None

        if image.shape[0]<image.shape[1]:
            pass
            image = cv2.resize(image, (1000,(int(image.shape[0] * 1000 / image.shape[1]))),
                               interpolation=cv2.INTER_AREA)
        else:
            image = rotate_about_center(image)
            image = cv2.resize(image, (1000, (int(image.shape[0] * 1000 / image.shape[1]))),
                               interpolation=cv2.INTER_AREA)

        # 灰度化并做加大对比度。
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # 获得梯度
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # 模糊
        blurred = cv2.blur(gradient, (4, 4))

        # 阈值
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 2*image.shape[0]-1,-40)

        # 腐蚀和膨胀数次, 移除噪点并让相邻的字符连接起来, 方便绘制包围盒。
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=10)

        try:
            rect = self._getBoundingRect(thresh)
        except cv2.error :
            return None

        # 将程序认定的算式的包围盒绘制出来, 并显示。
        # box = np.int0(cv2.boxPoints(rect))
        # cv2.drawContours(gray, [box], -1, (255, 255, 255), 3)
        # cv2.imshow('bounding box',gray)
        # cv2.waitKey(0)

        cutoff = self._crop(gray.copy(), rect)
        if type(cutoff) == 'Nonetype':
            return None
        cutoff = cv2.normalize(cutoff, None, 0, 255, cv2.NORM_MINMAX)

        # cv2.imshow('cut off', cutoff)
        # cv2.waitKey(0)

        cutoff = self._revert(cutoff)
        return cutoff


    def _do(self, filename=None,img=None,save_dir='./tmp1'):
        img = img = self._process(filename) if img==None else self._process(img=img)
        if img == None:
            return None

        # 获得轮廓
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25,-15)
        # print(img.shape)
        _, contours, __ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_list = []

        # 将获得的轮廓中, 面积太小的排除掉。并按照每个轮廓的质心的水平方向坐标进行排序,从而得到从左到右的序列。
        for con in contours:
            M = cv2.moments(con)

            if M['m00']/img.shape[0]<0.15:
                continue
            else:
                contours_list.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), con, M["m00"]])
        contours_list = sorted(contours_list, key=lambda x: x[0])

        # 用来在原图上绘制每个字符的包围盒的代码
        for i in contours_list:
            rect = cv2.minAreaRect(i[2])
            box = np.int0(cv2.boxPoints(rect))
            # draw a bounding box arounded the detected barcode and display the image
            cv2.drawContours(img, [box], -1, (255, 255, 255), 3)
        # cv2.imshow('bounding box',img)
        # cv2.waitKey(2000)

        # 处理每一个得到的包围盒,将原图分成许多小正方形, 可以选择保存或者展示或者传出。
        returnlist = []
        for ind, contour in enumerate(contours_list):
            tmp = self._extract(thresh, contour)
            tmp = self._resize_and_padding(tmp, (28, 28),ratio=1)
            returnlist.append(tmp)

            # 显示小图
            # cv2.imshow(str(ind), tmp)
            # cv2.waitKey(1000)

            # 存储小图
            # cv2.imwrite(save_dir+'/%d.bmp'% (self._getLastSaveFileIndex(save_dir)+1), tmp)
        # cv2.waitKey(0)
        return returnlist
        # return 'Found '+str(len(contours_list))+' signs.'


    # 获得某文件夹中所有以数字命名的文件中最大的数字
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
        return max(result)

    # 给定一个轮廓和原图, 将原图中轮廓内的部分裁剪出来。
    def _extract(self, img, packed_contour):
        mask = cv2.drawContours(np.zeros(img.shape), [packed_contour[2]], -1, 255, cv2.FILLED) * img
        Xs = [i[0][0] for i in packed_contour[2]]
        Ys = [i[0][1] for i in packed_contour[2]]
        x1 = min(Xs)
        x2 = max(Xs)
        y2 = max(Ys)
        y1 = min(Ys)
        hight = y2 - y1
        width = x2 - x1
        cropImg = mask[y1:y1+hight, x1:x1 + width]
        return cropImg

    # 将一个比例的尺寸不标准的图形, 填充成以size为大小, 以ratio为比例的图。
    def _resize_and_padding(self, img, size=(28, 28), ratio=1):
        a, b = img.shape
        c, d = 0, 0
        if ratio * a > b:
            c = ratio * a - b
        else:
            d = b / ratio - a
        img =  cv2.copyMakeBorder(img.copy(), int(d / 2), int(d - int(d / 2)), int(c / 2), int(c - int(c / 2)),
                                  cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    # 在相机拍摄的原图中, 寻找能够包围算式的包围盒, 并且自动略去一些小污点。
    def _getBoundingRect(self, img):
        (_, cnts, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        result = []
        for contour in cnts:
            M = cv2.moments(contour)
            if M['m00']<600:
                continue
            for j in contour:
                result.append(j[0])
        return cv2.minAreaRect(np.array(result))

    # 将包围算式的包围盒中的算式抠出来。
    def _crop(self, img, rect):
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

    # 反转颜色。
    def _revert(self, img):
        if img==None:
            return None
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                img[i, j] = 255 - img[i, j]
        return img





# 测试用代码
if __name__=='__main__':
    a = Preprocessor()
    a._do('./002.jpeg')
