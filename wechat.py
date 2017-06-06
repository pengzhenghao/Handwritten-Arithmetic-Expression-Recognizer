import itchat
from main import Main
import cv2
import re
main = Main()

# 一个小技巧处理了除号
def divide(str):
    str = str.replace(r'.-.',r'/')
    str = str.replace(r'-..',r'/')
    str = str.replace(r'..-',r'/')
    str = str.replace(r'--.',r'/')
    str = str.replace(r'.--',r'/')
    str = str.replace(r'--.',r'/')
    str = str.replace(r'---',r'/')
    return str

# 对私聊信息进行反馈
@itchat.msg_register(['Picture'])
def download_files(msg):
    msg['Text']('./tmp/'+msg['FileName'])

    # 进行训练数据的生成
    # st = main.processor.generateTrainData('./tmp/'+msg['FileName'],'./data/new_number/9')

    # 进行正式运行
    st = main.online_eval(cv2.imread('./tmp/' + msg['FileName']))

    try:
        st = divide(st)
        ans = eval(st)
    except SyntaxError:
        itchat.send( u'你写了"'+st+'"喵喵喵?' , msg['User']['UserName'])
        return
    itchat.send( st+'='+"%.5f"%ans, msg['User']['UserName'])
    return

# 对指定的群聊信息进行反馈
@itchat.msg_register(['Picture'],isGroupChat=True)
def download_files(msg):
    if msg['User']['NickName']!='SE' and  msg['User']['NickName']!=u'系统工程周四下午' and msg['User']['NickName']!=u'船舶与海洋工程-表情包交换群':
        return
    msg['Text']('./tmp/'+msg['FileName'])

    # 进行训练数据的生成
    # st = main.processor.generateTrainData('./tmp/'+msg['FileName'],'./data/new_number/9')

    # 进行正式运行
    st = main.online_eval(cv2.imread('./tmp/'+msg['FileName']))

    try:
        st = divide(st)
        ans = eval(st)
    except SyntaxError:
        itchat.send(   msg['User']['Self']['NickName']+u'写了"'+st+'"喵喵喵?' , msg['User']['UserName'])
        return
    itchat.send( msg['ActualNickName']+u'写了' +st+'='+"%.3f"%ans, msg['User']['UserName'])
    return

itchat.auto_login(hotReload=True)
itchat.run()