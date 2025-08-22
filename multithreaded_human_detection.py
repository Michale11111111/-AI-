import cv2
import numpy as np
import mss
from ultralytics import YOLO
import screeninfo
import ctypes
from ctypes import wintypes
import threading
import time

head_offsets_global = (0, 0) # 全局变量，用于存储头部偏移量
head_offsets_lock = threading.Lock() # 用于保护 head_offsets_global 的锁
mouse_move_lock = threading.Lock() # 用于保护鼠标移动的锁
# 加载 DLL
interception = ctypes.WinDLL(r".\x64\interception.dll")


# ----------------------------
# typedef 和 struct 定义
# ----------------------------

InterceptionContext = ctypes.c_void_p
InterceptionDevice = ctypes.c_int
InterceptionPrecedence = ctypes.c_int
InterceptionFilter = ctypes.c_ushort
InterceptionPredicate = ctypes.CFUNCTYPE(ctypes.c_int, InterceptionDevice)

# InterceptionMouseStroke
class InterceptionMouseStroke(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_ushort),
        ("flags", ctypes.c_ushort),
        ("rolling", ctypes.c_short),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("information", ctypes.c_uint),
    ]

# InterceptionKeyStroke
class InterceptionKeyStroke(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("state", ctypes.c_ushort),
        ("information", ctypes.c_uint),
    ]

# Stroke 定义为最大 size 的 union (用 byte array)
class InterceptionStroke(ctypes.Union):
    _fields_ = [
        ("mouse", InterceptionMouseStroke),
        ("key", InterceptionKeyStroke),
        ("data", ctypes.c_char * ctypes.sizeof(InterceptionMouseStroke)),
    ]


# ----------------------------
# 函数原型匹配
# ----------------------------

interception.interception_create_context.restype = InterceptionContext
interception.interception_destroy_context.argtypes = [InterceptionContext]
interception.interception_get_precedence.argtypes = [InterceptionContext, InterceptionDevice]
interception.interception_get_precedence.restype = InterceptionPrecedence
interception.interception_set_precedence.argtypes = [InterceptionContext, InterceptionDevice, InterceptionPrecedence]
interception.interception_get_filter.argtypes = [InterceptionContext, InterceptionDevice]
interception.interception_get_filter.restype = InterceptionFilter
interception.interception_set_filter.argtypes = [InterceptionContext, InterceptionPredicate, InterceptionFilter]
interception.interception_wait.argtypes = [InterceptionContext]
interception.interception_wait.restype = InterceptionDevice
interception.interception_wait_with_timeout.argtypes = [InterceptionContext, wintypes.ULONG]
interception.interception_wait_with_timeout.restype = InterceptionDevice
interception.interception_send.argtypes = [InterceptionContext, InterceptionDevice, ctypes.POINTER(InterceptionStroke), ctypes.c_uint]
interception.interception_send.restype = ctypes.c_int
interception.interception_receive.argtypes = [InterceptionContext, InterceptionDevice, ctypes.POINTER(InterceptionStroke), ctypes.c_uint]
interception.interception_receive.restype = ctypes.c_int
interception.interception_get_hardware_id.argtypes = [InterceptionContext, InterceptionDevice, ctypes.c_void_p, ctypes.c_uint]
interception.interception_get_hardware_id.restype = ctypes.c_uint
interception.interception_is_invalid.argtypes = [InterceptionDevice]
interception.interception_is_invalid.restype = ctypes.c_int
interception.interception_is_keyboard.argtypes = [InterceptionDevice]
interception.interception_is_keyboard.restype = ctypes.c_int
interception.interception_is_mouse.argtypes = [InterceptionDevice]
interception.interception_is_mouse.restype = ctypes.c_int
# 常量
INTERCEPTION_FILTER_MOUSE_ALL = 0xFFFF
INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN = 0x0040
INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN = 0x0001
INTERCEPTION_MOUSE_LEFT_BUTTON_UP   = 0x0002

# ----------------------------
# 回调函数
# ----------------------------
@InterceptionPredicate
def is_mouse(device):
    return interception.interception_is_mouse(device)

# ----------------------------
# 初始化函数
# ----------------------------
def init():
    ctx = interception.interception_create_context()
    if not ctx:
        raise RuntimeError("无法创建拦截上下文")
    interception.interception_set_filter(ctx, is_mouse, INTERCEPTION_FILTER_MOUSE_ALL)
    return ctx
def mouse_click(ctx, device):


    stroke = InterceptionStroke()
    stroke.mouse.state = INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN
    interception.interception_send(ctx, device, ctypes.byref(stroke), 1)
    stroke.mouse.state = INTERCEPTION_MOUSE_LEFT_BUTTON_UP
    interception.interception_send(ctx, device, ctypes.byref(stroke), 1)

model = YOLO("yolov8n-pose.pt")

COCO_KPTS = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
]
HEAD_PARTS = {"nose","left_eye","right_eye","left_ear","right_ear"}



# 截屏器
monitor_info = screeninfo.get_monitors()[0]
screen_width, screen_height = monitor_info.width, monitor_info.height
monitor = {"left": screen_width//2 - 150, "top": screen_height//2 - 150, "width": 300, "height": 300}

def detection_thread_func():
    global head_offsets_global
    sct = mss.mss() # 在线程内部初始化mss实例
    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]
        cx, cy = w/2, h/2
        current_head_offsets = []

        results = model.predict(frame, conf=0.35, verbose=False)
        for r in results:
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item()) if r.boxes.cls.cpu().numpy()[i] is not None else -1
                if cls_id != 0:
                    continue
                if r.keypoints is None: 
                    continue

                kxy = r.keypoints.xy[i].cpu().numpy()
                head_kpts = [kxy[COCO_KPTS.index(name)] for name in HEAD_PARTS if COCO_KPTS.index(name) < len(kxy) and kxy[COCO_KPTS.index(name)][0] > 0 and kxy[COCO_KPTS.index(name)][1] > 0]
                if head_kpts:
                    px, py = np.mean(head_kpts, axis=0)
                    dx, dy = px - cx, py - cy
                    current_head_offsets.append((dx, dy))
        if current_head_offsets:
            with head_offsets_lock:
                
                head_offsets_global = current_head_offsets[0] # 只取第一个检测到的头部偏移
                current_head_offsets[0]=(0,0)
        time.sleep(0.001) # 适当的延迟，避免CPU占用过高



if __name__ == "__main__":
    ctx = init()
    stroke = InterceptionStroke()

    # 启动检测线程
    detection_thread = threading.Thread(target=detection_thread_func, daemon=True)
    detection_thread.start()

    try:
        # PID 控制器参数
        Kp = 0.6 # 适当减小Kp，减少过冲
        Ki = 0.01
        Kd = 0.1 # 适当增加Kd，增加阻尼

        # 初始化PID状态变量
        prev_error_x = 0
        prev_error_y = 0
        integral_x = 0
        integral_y = 0
        prev_time = time.monotonic()

        while True:
            device = interception.interception_wait(ctx)
            if interception.interception_receive(ctx, device, ctypes.byref(stroke), 1) > 0:
                if interception.interception_is_mouse(device):
                    mouse = stroke.mouse
                    if mouse.state==INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN and head_offsets_global!=(0,0) :
                        with head_offsets_lock:
                            dx, dy = head_offsets_global # 从全局变量获取最新的偏移量
                        print("dx:",dx)
                        print("dy:",dy)

                        # 获取当前鼠标坐标
                        # 鼠标移动，考虑灵敏度调整和相对移动
                        # 这里的 dx, dy 是相对于 300x300 区域中心的偏移
                        # 假设需要将这个偏移量映射到屏幕上的实际移动距离
                        # 可以尝试一个缩放因子，例如 0.5 或 1.0，根据实际效果调整
                        sensitivity = 1.5 # 调整灵敏度
                        # 鼠标使用后清零偏移量，避免重复移动或旧数据干扰
                        

                        # 误差
                        error_x = dx * sensitivity
                        error_y = dy * sensitivity

                        # 基于时间步长的 PID 计算
                        now = time.monotonic()
                        dt = max(1e-3, now - prev_time)

                        # 积分项（带防积分饱和）
                        integral_x += error_x * dt
                        integral_y += error_y * dt
                        integral_limit = 100.0
                        if integral_x > integral_limit:
                            integral_x = integral_limit
                        elif integral_x < -integral_limit:
                            integral_x = -integral_limit
                        if integral_y > integral_limit:
                            integral_y = integral_limit
                        elif integral_y < -integral_limit:
                            integral_y = -integral_limit

                        # 微分项（带时间）
                        derivative_x = (error_x - prev_error_x) / dt
                        derivative_y = (error_y - prev_error_y) / dt

                        # PID 输出
                        output_x = Kp * error_x + Ki * integral_x + Kd * derivative_x
                        output_y = Kp * error_y + Ki * integral_y + Kd * derivative_y

                        with mouse_move_lock:
                            for i in range(0,max(abs(int(output_x+8*output_x/100)),abs(int(output_y+output_y/100*8)))):
                                if(i<=abs(int(output_x+8*output_x/100))):
                                    stroke.mouse.x = int(1*np.sign(output_x))
                                else:
                                    stroke.mouse.x=0

                                if(i<=abs(int(output_y+output_y/100*8))):
                                    stroke.mouse.y = int(1*np.sign(output_y))
                                else:
                                    stroke.mouse.y=0
                                if(i!=max(abs(int(output_x+8*output_x/100)),abs(int(output_y+output_y/100*8)))-1):
                                    stroke.mouse.state=0x0000
                                else:
                                    stroke.mouse.state=INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN
                                    interception.interception_send(ctx, device, ctypes.byref(stroke), 1)
                                    stroke.mouse.state=INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN
                                    stroke.mouse.x=0
                                    stroke.mouse.y=0
                                    interception.interception_send(ctx, device, ctypes.byref(stroke), 1)
                                    break
                                
                                interception.interception_send(ctx, device, ctypes.byref(stroke), 1)
                            
                        # 更新 PID 状态
                        prev_error_x = error_x
                        prev_error_y = error_y
                        prev_time = now
                        interception.interception_send(ctx, device, ctypes.byref(stroke), 1)
                    head_offsets_global = (0, 0)

            
            interception.interception_send(ctx, device, ctypes.byref(stroke), 1)

    except KeyboardInterrupt:
        print("退出程序")
    finally:
        interception.interception_destroy_context(ctx)