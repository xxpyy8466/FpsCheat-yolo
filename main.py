import math
import threading
import time
# import pyautogui
import sys
import signal
import random

import numpy as np
import torch
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import smart_inference_mode

from ScreenShot import screenshot
import cv2,time,win32print,win32con,win32gui
import pynput.mouse
from pynput.mouse import Listener
from pynput import keyboard
import a
from SendInput import *
import pyautogui


def signal_handler(sig, frame):
    print('退出')
    # 在这里添加你想要做的清理操作
    # 例如停止子进程，关闭文件等
    # ...
    # 退出程序的代码
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

is_x2_pressed = False


def mouse_click(x, y, button, pressed):
    global is_x2_pressed
    # print("debug")
    # print(x, y, button, pressed)
    if pressed and button == pynput.keyboard.Key:
        is_x2_pressed = True
    elif not pressed and button == pynput.mouse.Button.right:
        is_x2_pressed = False


def mouse_listener():
    with Listener(on_click=mouse_click) as listener:
        listener.join()

mode=0
global h

def on_activate_f1():
    print('暂停检测')
    global mode
    mode=1

def on_activate_f2():
    print('开启ju模式')
    global mode
    mode=2

def on_activate_f3():
    print('开启bu模式')
    global mode
    mode=3

def on_activate_f4():
    print('开启ls模式')
    global mode
    mode=4

def on_activate_f5():
    print('开启qbz模式')
    global mode
    mode=5

def on_activate_alt_c():
    print('<alt>+c pressed')
    h.stop()


h=keyboard.GlobalHotKeys({
        '<f1>': on_activate_f1,
        '<f2>': on_activate_f2,
        '<f3>': on_activate_f3,
        '<f4>':  on_activate_f4,
        '<f5>':  on_activate_f5,
        '<alt>+c':on_activate_alt_c})
h.start()


def on_press(key):
    if key == keyboard.Key.shift_l:
        print(6666)
        global is_x2_pressed
        is_x2_pressed=True

def on_release(key):
    if key == keyboard.Key.shift_l:
        global is_x2_pressed
        is_x2_pressed=False

def listen_key():
    with keyboard.Listener(
    on_press=on_press,
    on_release=on_release) as listener:
        listener.join()

Detect = 640
# 获取真实的分辨率
ScreenX = win32print.GetDeviceCaps(win32gui.GetDC(0), win32con.DESKTOPHORZRES)
ScreenY = win32print.GetDeviceCaps(win32gui.GetDC(0), win32con.DESKTOPVERTRES)
# 以中心为单位，向左上角偏移320
XLeft = int((ScreenX / 2) - (Detect / 2))
YLeft = int((ScreenY / 2) - (Detect / 2))
XRight = XLeft + Detect
YRight = YLeft + Detect
# # 中心点（目标点-中心点使用）
CoreX = ((XRight - XLeft) / 2)
CoreY = ((YRight - YLeft) / 2)
print(ScreenX,ScreenY,XLeft,YLeft,XRight,YRight,CoreX,CoreY)

@smart_inference_mode()
def run():
    global is_x2_pressed
    # Load model
    # device = torch.device('cuda:0')
    # model = DetectMultiBackend(weights='./weights/yolov5n.pt', device=device, dnn=False, data=False, fp16=True)
    device = torch.device('cpu')
    model = DetectMultiBackend(weights='./weights/Valorant.pt', device=device, dnn=False, data=False, fp16=False)

    obj = a.RunLogitechTwo()

    # 读取图片
    while True:
        im = screenshot()

        im0 = im

        # 处理图片
        im = letterbox(im, (640, 640), stride=32, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # 推理
        pred = model(im, augment=False, visualize=False)
        # 非极大值抑制  classes=0 只检测人
        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=(0,1), max_det=3)[0]
        if not len(pred):
        # print("未检测到目标")
            pass
        dis_list = []
        target_list = []
        for *xyxy, scores, labels in reversed(pred):  # 处理推理出来每个目标的信息
            #用map函数将x1,y1,x2,y2转换为round类型
            x1,y1,x2,y2 = map(round,(torch.tensor(xyxy).view(1, 4).view(-1).tolist()))
            x,y,w,h = map(round,((xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()))
            scores = float(scores)
            if int(labels) == 0 or int(labels) == 1:
                #计算距离
                dis = math.sqrt(math.pow(x - CoreX, 2) + \
                    math.pow(y - CoreY, 2))
                dis_list.append(dis)
                target_list.append([x, y, w, h])
        if len(dis_list) != 0 and is_x2_pressed:
            x,y,w,h = target_list[dis_list.index(min(dis_list))]
            print("目标信号坐标：",x,y,w,h)
            multiple_x,multiple_y=1,1
            if mode==2:
                multiple_x,multiple_y=1.8,1
            x1 = round((int(x)-CoreX)*multiple_x)
            y1 = round((int(y)-CoreY)*multiple_y)
            print(x1,y1)
            obj.move_xy(x1,y1)
            time.sleep(0.3)
            obj.lei_shen()
                # time.sleep(1)  # 主动睡眠，防止推理过快,鼠标移动相同的两次       
        # # 处理推理内容
        # for i, det in enumerate(pred):
        #     # 画框
        #     # annotator = Annotator(im0, line_width=2)
        #     if len(det):
        #         distance_list = []  # 距离列表
        #         target_list = []  # 敌人列表
        #         # 将转换后的图片画框结果转换成原图上的结果
        #         det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        #         for *xyxy, conf, cls in reversed(det):  # 处理推理出来每个目标的信息
        #             # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
        #             # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
        #             x,y,w,h = map(round,((xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()))

        #             # 鼠标移动值
        #             # X = xywh[0] - 320
        #             # Y = xywh[1] - 320

        #             # distance = math.sqrt(X ** 2 + Y ** 2)  # 鼠标距离敌人距离 勾股
        #             distance=math.sqrt(math.pow(x - CoreX, 2) + \
        #                 math.pow(y - CoreY, 2))
        #             # xywh.append(distance)
        #             # annotator.box_label(xyxy, label=f'[{int(cls)}Distance:{round(distance, 2)}]',  # 框上显示距离
        #             #                     color=(34, 139, 34),
        #             #                     txt_color=(0, 191, 255))

        #             distance_list.append(distance)
        #             target_list.append([x, y, w, h])

        #         # 鼠标移动值 获取距离最小的目标
        #         # target_info = target_list[distance_list.index(min(distance_list))]
        #         # print(f"目标信息：{target_info}")
        #         x,y,w,h = target_list[distance_list.index(min(distance_list))]
        #         print("目标信号坐标：",x,y,w,h)
        #         x1 = round(x-CoreX)
        #         y1 = round(y-CoreY)
        #         print(x1,y1)

        #         if is_x2_pressed:
        #             # target_x = (1920 / 2) + int(target_info[0] - 320)
        #             # target_y = (1080 / 2) + int(target_info[1] - 320)
        #             # obj.quick_move(target_x,target_y)
        #             # obj.quick_move(int(X),int(Y))
        #             obj.move_xy(x1,y1)
        #         # pyautogui.moveTo(target_x, target_y)
        #         # time.sleep(0.03)  # 主动睡眠，防止推理过快,鼠标移动相同的两次

        #     # im0 = annotator.result()
        #     # cv2.imshow('window', im0)
            cv2.waitKey(1)


if __name__ == "__main__":
    # threading.Thread(target=mouse_listener).start()
    listen_key_thread=threading.Thread(target=listen_key)
    listen_key_thread.daemon=True
    listen_key_thread.start()
    run()
