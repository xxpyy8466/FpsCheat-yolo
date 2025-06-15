import ctypes
import os
import sys
from time import sleep

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


try:
    # 获取当前绝对路径
    root = os.path.abspath(os.path.dirname(__file__))
    driver = ctypes.CDLL('C:/cfwzw/logitech.driver.dll')
    ok = driver.device_open() == 1  # 该驱动每个进程可打开一个实例
    if not ok:
        print('错误, GHUB驱动没有找到')

    # if getattr(sys, 'frozen', None):
    #     basedir = sys._MEIPASS
    # else:
    #     basedir = os.path.dirname(__file__)

    # with suppress_stdout_stderr():
    #     path=os.path.join(basedir,'logitech.driver.dll')
    #     driver = ctypes.CDLL(path)
    #     ok = driver.device_open() == 1  # 该驱动每个进程可打开一个实例
    #     if not ok:
    #         print('错误, GHUB驱动没有找到')
except FileNotFoundError:
    print(f'错误, DLL 文件没有找到')

sys.stdout = sys.__stdout__
print('启动成功!')

class Logitech:
    class mouse:

        """
        code: 1:左键, 2:中键, 3:右键
        """

        @staticmethod
        def press(code):
            if not ok:
                return
            driver.mouse_down(code)

        @staticmethod
        def release(code):
            if not ok:
                return
            driver.mouse_up(code)

        @staticmethod
        def click(code):
            if not ok:
                return
            driver.mouse_down(code)
            driver.mouse_up(code)

        @staticmethod
        def scroll(a):
            """
            鼠标滚轮
            """
            if not ok:
                return
            driver.scroll(a)

        @staticmethod
        def move(x, y):
            """
            相对移动, 绝对移动需配合 pywin32 的 win32gui 中的 GetCursorPos 计算位置
            pip install pywin32 -i https://pypi.tuna.tsinghua.edu.cn/simple
            x: 水平移动的方向和距离, 正数向右, 负数向左
            y: 垂直移动的方向和距离
            """
            if not ok:
                return
            if x == 0 and y == 0:
                return
            driver.moveR(x, y, False)

    class keyboard:

        """
        键盘按键函数中，传入的参数采用的是键盘按键对应的键码
        code: 'a'-'z':A键-Z键, '0'-'9':0-9
        """

        @staticmethod
        def press(code):

            if not ok:
                return
            driver.key_down(code)

        @staticmethod
        def release(code):
            if not ok:
                return
            driver.key_up(code)

        @staticmethod
        def click(code):
            if not ok:
                return
            driver.key_down(code)
            driver.key_up(code)


class RunLogitechTwo:
    def __init__(self):
        self.log_mouse = Logitech.mouse
        self.log_keyboard = Logitech.keyboard
        pass

    def quick_move(self):
        # time.sleep(random.randint(1, 3))
        self.log_mouse.click(3)
        sleep(0.05)
        self.log_mouse.click(1)
        # print('hahaha')

    def move_xy(self,x,y):
        # time.sleep(random.randint(1, 3))
        #self.log_mouse.click(3)
        #sleep(0.05)
        # self.log_mouse.click(1)
        print(int(x),int(y))
        self.log_mouse.move(int(x),int(y))
        # print('hahaha')

    def shun_ju(self):
        # time.sleep(random.randint(1, 3))
        self.log_mouse.click(3)
        sleep(0.075)
        self.log_mouse.click(1)
        sleep(0.12)
       # print('hahaha')

    def lei_shen(self):
        # time.sleep(random.randint(1, 3))
        self.log_mouse.press(1)
        sleep(0.125)
        self.log_mouse.click(1)
        self.log_mouse.press(1)
        sleep(0.125)
        self.log_mouse.click(1)

    def qbz(self):
        # time.sleep(random.randint(1, 3))
        self.log_mouse.press(1)
        sleep(0.155)
        self.log_mouse.click(1)

    def keyboard_click(self):
        self.log_keyboard.click('q')
        sleep(0.11)
        self.log_keyboard.click('q')


