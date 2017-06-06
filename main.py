# 本类是本程序的主入口。提供实时识别的服务。
from preprocess import Preprocessor
from network import Network
import numpy as np
import cv2

class Main(object):

    def __init__(self):
        self.processor = Preprocessor()
        self.network = Network()
        self.dic = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'+',11:'-',12:'*',13:'.'}

    # 输入图像文件的地址, 返回结果, 一般用于测试。
    def eval(self,file_dir):
        img_list = self.processor.generateTrainData(file_dir)
        result = []
        for i in img_list:
            inp = np.array( i )
            inp = np.resize(inp,[1,784])
            result.append(self.network.eval(inp)[0])
        print(result)
        ans = ''
        for i in range(len(result)):
            ans = ans + self.dic[result[i]]
        return ans

    # 输入图像文件, 返回结果, 一般用于实际应用中: 从摄像机、微信获得图片。
    def online_eval(self,img):
        try:
            img_list = self.processor.onlineTest(img=img)
        except cv2.error:
            return '0'
        if img_list == None:
            return '0'
        result = []
        for i in img_list:
            inp = np.array( i )
            inp = np.resize(inp,[1,784])
            result.append(self.network.eval(inp)[0])
        ans = ''
        for i in range(len(result)):
            ans = ans + self.dic[result[i]]
        return ans


# For test
if __name__=='__main__':
    main = Main()
    main.eval('002.jpeg')