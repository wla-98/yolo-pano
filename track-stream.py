import cv2 
import numpy as np 
from ultralytics import YOLO 
import time 
import torch 
from deep_sort_realtime.deepsort_tracker  import DeepSort 
from collections import deque 
import socket 
import struct 
import threading 
import subprocess 
import logging 
from http.server  import BaseHTTPRequestHandler, HTTPServer 
import json 
import queue 
from dataclasses import dataclass 
from scipy.interpolate  import make_interp_spline  # Add this import at the top of your file 

# 增强日志配置 
logging.basicConfig(  
    level=logging.INFO,
    format='%(asctime)s [%(threadName)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log') 
    ]
)
logger = logging.getLogger(__name__)  

# 控制输出平滑处理
def smooth_control(current_control, history):
    history.append(current_control)
    if len(history) < 2:
        return current_control
    # 使用加权平均平滑控制输出
    weights = np.linspace(0.5, 1.0, len(history))
    weights /= weights.sum()
    smoothed = np.average(list(history), weights=weights)
    return smoothed 

# 全局配置 
@dataclass 
class Config:
    RTSP_INPUT_URL = 'rtsp://192.168.1.22:554/ch01_sub'
    RTSP_OUTPUT_URL = "rtsp://192.168.1.193:25544/yolo"
    HTTP_SERVER_PORT = 8080 
    CONTROL_IP = "192.168.1.120"
    CONTROL_PORT = 43893 
    LOCAL_PORT = 20001 
    MAX_FRAME_QUEUE_SIZE = 5 
    MAX_CONTROL_QUEUE_SIZE = 10 
 
# 设备设置 
device = 'cuda' if torch.cuda.is_available()  else 'cpu'
logger.info(f"Using  device: {device}")
 
# 全局共享资源（增强线程安全）
class SharedState:
    def __init__(self):
        self.frame_queue  = queue.Queue(maxsize=Config.MAX_FRAME_QUEUE_SIZE)
        self.processed_frame  = None 
        self.frame_lock  = threading.RLock()  # 改为可重入锁 
        self.control_queue  = queue.Queue(maxsize=Config.MAX_CONTROL_QUEUE_SIZE)
        self.tracking_state  = TrackingState()
        self.stream_active  = True 
        self.detection_active  = True 
        self.frame_dimensions  = (None, None)
        self.fps  = None 
        self.state_lock  = threading.Lock()  # 用于状态变量的锁 
 
# 增强型PID控制器类(输出为0.0-1.0 rad/s的角速度)
class EnhancedPIDController:
    def __init__(self, kp, ki, kd, max_speed=1.0, dead_zone=5, smooth_factor=0.2):
        """
        增强型PID控制器(输出角速度)

        参数:
        - kp: 比例增益
        - ki: 积分增益
        - kd: 微分增益
        - max_speed: 最大输出角速度(rad/s)
        - dead_zone: 死区大小(像素)
        - smooth_factor: 误差平滑因子(0-1)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_speed = max_speed
        self.dead_zone = dead_zone
        self.smooth_factor = smooth_factor

        # 状态变量
        self.prev_error = 0
        self.prev_error_smooth = 0
        self.integral = 0
        self.integral_limit = max_speed * 0.5  # 积分限幅
        self.error_history = deque(maxlen=5)  # 用于平滑误差

    def update(self, error, dt=0.033):
        """
        更新PID控制器

        参数:
        - error: 当前误差(像素)
        - dt: 时间步长(秒)

        返回:
        - 角速度控制输出(rad/s)
        """
        # 死区处理
        if abs(error) < self.dead_zone:
            error = 0

        # 误差平滑滤波
        self.error_history.append(error)
        smooth_error = np.mean(self.error_history) if self.error_history else error
        smooth_error = self.smooth_factor * smooth_error + (1 - self.smooth_factor) * self.prev_error_smooth

        # 计算微分项(使用平滑后的误差)
        derivative = (smooth_error - self.prev_error_smooth) / max(dt, 0.001)  # 防止除以0

        # 计算积分项(只在误差较大时积分)
        if abs(smooth_error) > self.dead_zone * 0.5:
            self.integral += smooth_error * dt
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        else:
            self.integral *= 0.9  # 缓慢释放积分

        # 计算PID输出(转换为角速度)
        output = self.kp * smooth_error + self.ki * self.integral + self.kd * derivative

        # 输出限幅到[0.0, max_speed]范围
        output = np.clip(output, -self.max_speed, self.max_speed)

        # 更新状态
        self.prev_error = error
        self.prev_error_smooth = smooth_error

        return output

    def dynamic_adjust(self, error_magnitude):
        """
        根据误差大小动态调整PID参数

        参数:
        - error_magnitude: 误差大小(绝对值)
        """
        # 误差较大时增加P增益，减小I增益
        if error_magnitude > 100:
            self.kp = 0.002  # 减小比例增益，防止超调
            self.ki = 0.0001  # 减小积分增益
            self.kd = 0.001   # 适当增加微分增益
        # 中等误差时保持默认参数
        elif error_magnitude > 30:
            self.kp = 0.0015
            self.ki = 0.0002
            self.kd = 0.0008
        # 小误差时减小P增益，增加I增益
        else:
            self.kp = 0.001
            self.ki = 0.0003
            self.kd = 0.0005


class RobotCommander:
    def __init__(self, local_port=20001, ctrl_ip="192.168.1.120", ctrl_port=43893, angle_port=54321):
        self.local_port = local_port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.ctrl_addr = (ctrl_ip, ctrl_port)

        # 设置角度接收socket
        # self.angle_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.angle_socket.bind(('0.0.0.0', angle_port))
        # 当前角度值
        # self.current_angle = 0.0
        # 启动角度接收线程
        # self.angle_thread = threading.Thread(target=self.receive_angles, daemon=True)
        # self.angle_thread.start()

    def receive_angles(self):
        while True:
            try:
                data, addr = self.angle_socket.recvfrom(1024)
                radians = float(data.decode('utf-8'))
                self.current_angle = radians

                # 预设角速度 (rad/s) - 可根据需要调整
                angular_velocity = 1.098  # 1.12 rad/s

                # 计算需要发送指令的持续时间
                duration = abs(radians) / angular_velocity

                # 将弧度值转换为机器人控制参数
                # 根据角度方向决定参数符号
                param_sign = 1 if radians >= 0 else -1
                param = 32767 * param_sign  # 示例转换，可能需要调整

                # 计算发送次数 (假设每50ms发送一次指令)
                interval = 0.035  # 35ms
                iterations = int(duration / interval)

                print(f"开始旋转: 角度={radians:.2f}rad, 角速度={angular_velocity}rad/s, 持续时间={duration:.2f}s, 发送次数={iterations}")

                # 在持续时间内持续发送指令
                for i in range(iterations):
                    self.send_command(0x21010135, param)
                    time.sleep(interval)

            except Exception as e:
                print(f"接收角度错误: {e}")

    def send_command(self, code, param1=0, param2=0):
        print(f"发送命令：Code={code}, Param1={param1}, Param2={param2}")
        self.send_simple(code, param1, param2)

    def send_simple(self, code, param1=0, param2=0):
        try:
            payload = struct.pack('<3i', code, param1, param2)
            self.server.sendto(payload, self.ctrl_addr)
        except Exception as e:
            print(f"发送命令时出错：{e}")

 
# 跟踪状态（增加调试日志）
class TrackingState:
    def __init__(self):
        self.tracker  = DeepSort(max_age=3, n_init=1)
        self.target_id  = None 
        self.track_history  = deque(maxlen=20)
        self.tracking_start_time  = None 
        self.tracking_lost_time  = None 
        self.is_tracking  = False 
        self.last_detection_time  = 0 
        self.detection_interval  = 0.0 
        self.tracking_interval  = 0.1 
        self.last_control_time  = time.time()   
        self.last_target_center  = None 
        self.control_history  = deque(maxlen=5)
        self.target_coords  = None 
        self.clicked_point  = None 
        self.lock  = threading.RLock()
 
    def reset(self):
        with self.lock: 
            logger.info("Resetting  tracking state")
            self.tracker  = DeepSort(max_age=3, n_init=1)
            self.target_id  = None 
            self.track_history.clear()   
            self.tracking_start_time  = None 
            self.tracking_lost_time  = None 
            self.is_tracking  = False 
            self.last_target_center  = None 
            self.target_coords  = None 
            self.clicked_point  = None 
 
# HTTP处理器（优化响应速度和资源释放）
class HTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info("%s  - - [%s] %s" % (self.address_string(), 
                                        self.log_date_time_string(), 
                                        format%args))
    
    def do_POST(self):
        global shared_state 
        
        try:
            # 记录请求开始 
            logger.info(f"Handling  POST request from {self.client_address}") 
            
            # 1. 验证内容长度 
            if 'Content-Length' not in self.headers:  
                self.send_error(411,  "Content-Length header required")
                return 
                
            # 2. 验证内容长度合理性 
            try:
                content_length = int(self.headers['Content-Length']) 
            except ValueError:
                self.send_error(400,  "Invalid Content-Length")
                return 
                
            if content_length <= 0 or content_length > 1024:
                self.send_error(413  if content_length > 1024 else 400, 
                              "Payload too large (max 1KB)" if content_length > 1024 else "Invalid Content-Length")
                return 
                
            # 3. 读取和解析JSON数据（带超时）
            start_time = time.time() 
            post_data = b''
            while len(post_data) < content_length and time.time()  - start_time < 1.0:
                post_data += self.rfile.read(min(1024,  content_length - len(post_data)))
            
            try:
                data = json.loads(post_data.decode('utf-8'))  
            except json.JSONDecodeError as e:
                response = {
                    "status": "Error",
                    "message": f"Invalid JSON format: {str(e)}",
                    "example": {"x": 0.5, "y": 0.5}
                }
                self._send_response(400, response)
                return 
            except UnicodeDecodeError:
                self._send_response(400, {"status": "Error", "message": "Invalid character encoding (UTF-8 required)"})
                return 
                
            # 4. 验证必需字段 
            if 'x' not in data or 'y' not in data:
                self._send_response(400, {
                    "status": "Error",
                    "message": "Both x and y coordinates are required",
                    "required_fields": ["x", "y"],
                    "example": {"x": 0.5, "y": 0.5}
                })
                return 
                
            # 5. 验证坐标范围 
            try:
                x_norm = float(data['x'])
                y_norm = float(data['y'])
            except (TypeError, ValueError):
                self._send_response(400, {
                    "status": "Error",
                    "message": "Coordinates must be numbers",
                    "received": {"x": data['x'], "y": data['y']},
                    "example": {"x": 0.5, "y": 0.5}
                })
                return 
                
            if (x_norm != -1 or y_norm != -1) and (x_norm < 0 or x_norm > 1 or y_norm < 0 or y_norm > 1):
                self._send_response(400, {
                    "status": "Error",
                    "message": "Coordinates must be between 0 and 1 (or -1,-1 to reset)",
                    "received_coordinates": (x_norm, y_norm),
                    "valid_range": "[0-1] for tracking, (-1,-1) to reset"
                })
                return 
                
            # 6. 处理跟踪逻辑 
            response = self._handle_tracking_logic(x_norm, y_norm)
            
            # 7. 发送响应 
            self._send_response(200, response)
            
        except Exception as e:
            logger.error(f"Unexpected  error in HTTP handler: {str(e)}", exc_info=True)
            self._send_response(500, {
                "status": "Error",
                "message": "Internal server error",
                "error": str(e)
            })
    
    def _handle_tracking_logic(self, x_norm, y_norm):
        global shared_state 
        
        with shared_state.tracking_state.lock: 
            ts = shared_state.tracking_state   
            current_time = time.time()  
            
            # 重置命令 
            if x_norm == -1 and y_norm == -1:
                ts.reset()  
                return {
                    "status": "Success",
                    "message": "Tracking reset",
                    "mode": "Detection",
                    "timestamp": current_time 
                }
            
            # 已在跟踪中 - 忽略新坐标 
            elif ts.is_tracking:  
                return {
                    "status": "Info",
                    "message": "Currently tracking, new coordinates ignored. Send (-1,-1) to reset.",
                    "mode": "Tracking",
                    "current_target": ts.target_id,  
                    "timestamp": current_time 
                }
            
            # 开始新跟踪 
            else:
                if None in shared_state.frame_dimensions:  
                    return {
                        "status": "Error",
                        "message": "Video stream not initialized",
                        "timestamp": current_time 
                    }
                
                width, height = shared_state.frame_dimensions   
                x_pixel = int(x_norm * width)
                y_pixel = int(y_norm * height)
                
                ts.target_coords  = (x_pixel, y_pixel)
                ts.is_tracking  = True 
                ts.tracking_start_time  = current_time 
                
                logger.info(f"Starting  new tracking at pixel coordinates: ({x_pixel}, {y_pixel})")
                
                return {
                    "status": "Success",
                    "message": "Tracking started",
                    "normalized_coordinates": (x_norm, y_norm),
                    "pixel_coordinates": (x_pixel, y_pixel),
                    "frame_dimensions": shared_state.frame_dimensions,  
                    "mode": "Tracking",
                    "timestamp": current_time 
                }
    
    def _send_response(self, code, data):
        """封装响应发送逻辑"""
        try:
            response = json.dumps(data).encode('utf-8') 
            self.send_response(code) 
            self.send_header('Content-type',  'application/json')
            self.send_header('Content-Length',  str(len(response)))
            self.end_headers() 
            self.wfile.write(response) 
        except Exception as e:
            logger.error(f"Failed  to send response: {e}")
 
def start_http_server():
    server_address = ('0.0.0.0', Config.HTTP_SERVER_PORT)
    httpd = HTTPServer(server_address, HTTPRequestHandler)
    httpd.timeout  = 1  # 设置超时避免阻塞 
    logger.info(f"Starting  HTTP server on port {Config.HTTP_SERVER_PORT}...")
    
    while getattr(httpd, '_shutdown_request', False) is False:
        httpd.handle_request() 
    
    logger.info("HTTP  server shutdown")
 
# 视频捕获线程（增强重连逻辑）
class VideoCaptureThread(threading.Thread):
    def __init__(self, shared_state):
        super().__init__(name="VideoCaptureThread")
        self.shared_state  = shared_state 
        self.running  = True 
        self.cap  = None 
        self.retry_interval  = 5 
        self.max_retries  = 10 
        self.last_frame_time  = 0 
 
    def run(self):
        retry_count = 0 
        last_log_time = 0 
        frame_count = 0 
        
        while self.running  and retry_count < self.max_retries: 
            try:
                if self.cap  is None or not self.cap.isOpened(): 
                    self._reconnect_stream()
                    retry_count = 0 
                
                ret, frame = self.cap.read()  
                if not ret:
                    logger.warning("Frame  read failed, attempting to reconnect...")
                    self._cleanup_capture()
                    retry_count += 1 
                    time.sleep(self.retry_interval) 
                    continue 
                
                # 计算FPS 
                frame_count += 1 
                current_time = time.time() 
                if current_time - last_log_time > 5:
                    fps = frame_count / (current_time - last_log_time)
                    logger.info(f"Capture  FPS: {fps:.1f}")
                    frame_count = 0 
                    last_log_time = current_time 
                
                # 放入队列（非阻塞方式）
                try:
                    if self.shared_state.frame_queue.full(): 
                        try:
                            self.shared_state.frame_queue.get_nowait() 
                        except queue.Empty:
                            pass 
                    
                    self.shared_state.frame_queue.put(frame.copy(),  timeout=0.1)
                    self.last_frame_time  = time.time() 
                    
                except queue.Full:
                    logger.warning("Frame  queue full, dropping frame")
                
            except Exception as e:
                logger.error(f"Video  capture error: {e}")
                self._cleanup_capture()
                retry_count += 1 
                time.sleep(self.retry_interval) 
        
        if retry_count >= self.max_retries: 
            logger.error("Max  retries reached for video capture")
            with self.shared_state.state_lock: 
                self.shared_state.stream_active  = False 
                self.shared_state.detection_active  = False 
 
    def _reconnect_stream(self):
        """重新连接视频流"""
        logger.info(f"Connecting  to RTSP stream: {Config.RTSP_INPUT_URL}")
        
        # 移除 setExceptionMode 调用 
        self.cap  = cv2.VideoCapture(Config.RTSP_INPUT_URL)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,  1)
        self.cap.set(cv2.CAP_PROP_FPS,  15)
        
        if not self.cap.isOpened(): 
            raise RuntimeError("Failed to open RTSP stream")
        
        # 获取流属性 
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        fps = self.cap.get(cv2.CAP_PROP_FPS) 
        
        if fps <= 0 or fps > 60:
            fps = 25 
            logger.warning(f"Invalid  FPS detected, using default {fps}")
        
        with self.shared_state.state_lock: 
            self.shared_state.frame_dimensions  = (width, height)
            self.shared_state.fps  = fps 
        
        logger.info(f"Stream  resolution: {width}x{height} @ {fps:.1f} fps")
 
    def _cleanup_capture(self):
        """清理捕获资源"""
        if self.cap  is not None:
            try:
                self.cap.release() 
            except:
                pass 
            self.cap  = None 
 
    def stop(self):
        """安全停止线程"""
        self.running  = False 
        self._cleanup_capture()
 
# 检测和跟踪线程（优化处理逻辑）
class DetectionTrackingThread(threading.Thread):
    def __init__(self, shared_state, model, yaw_pid, pitch_pid):
        super().__init__(name="DetectionTrackingThread")
        self.shared_state  = shared_state 
        self.model  = model 
        self.yaw_pid  = yaw_pid 
        self.pitch_pid  = pitch_pid 
        self.running  = True 
        self.commander  = RobotCommander()
        self.last_processed_time  = 0 
        
        # 初始化平滑相关的属性 
        self.smoothing_factor  = 0.2  # 平滑系数 
        self.smoothed_center  = None  # 平滑后的中心点 
        self.track_history  = deque(maxlen=10)  # 跟踪历史记录 
 
 
    def run(self):
        frame_count = 0 
        start_time = time.time()  
        
        while self.running  and self.shared_state.detection_active: 
            try:
                # 控制处理频率 
                current_time = time.time() 
                if current_time - self.last_processed_time  < 0.033:  # ~30fps 
                    time.sleep(0.001) 
                    continue 
                
                self.last_processed_time  = current_time 
                
                # 从队列获取最新帧（带超时）
                try:
                    frame = self.shared_state.frame_queue.get(timeout=0.5) 
                except queue.Empty:
                    logger.debug("Frame  queue empty, skipping...")
                    continue 
                
                # 处理帧 
                processed_frame = self.process_frame(frame) 
                
                # 更新处理后的帧 
                with self.shared_state.frame_lock: 
                    self.shared_state.processed_frame  = processed_frame 
                
                # FPS计算 
                frame_count += 1 
                if frame_count % 100 == 0:
                    fps = 100 / (time.time()  - start_time)
                    logger.info(f"Processing  FPS: {fps:.1f}")
                    start_time = time.time() 
                
            except Exception as e:
                logger.error(f"Detection/tracking  error: {e}")
                time.sleep(0.1) 
        
        logger.info("Detection/tracking  thread stopped")
 
    def process_frame(self, frame):
        """处理帧的检测和跟踪逻辑"""
        height, width, _ = frame.shape    
        center_x, center_y = width // 2, height // 2 
        
        # 绘制中心十字线 
        cv2.line(frame,  (center_x-10, center_y), (center_x+10, center_y), (0,255,0), 2)
        cv2.line(frame,  (center_x, center_y-10), (center_x, center_y+10), (0,255,0), 2)
        
        # 获取跟踪状态 
        with self.shared_state.tracking_state.lock:  
            ts = self.shared_state.tracking_state    
            current_time = time.time()   
            
            if not ts.is_tracking:  
                # 检测模式 
                self._handle_detection_mode(frame, ts, current_time)
            else:
                # 跟踪模式 
                dt = current_time - ts.last_control_time   
                ts.last_control_time  = current_time 
                
                # 优先处理手动指定的目标 
                found_target = self._handle_manual_tracking(frame, ts) if ts.target_coords  else False 
                
                # 自动目标跟踪 
                if not found_target:
                    found_target = self._handle_automatic_tracking(frame, ts, current_time, dt)
                
                # 目标丢失处理 
                if not found_target:
                    self._handle_tracking_lost(ts, current_time)
            
            # 更新模式指示器 
            self._update_mode_indicator(frame, width, height, ts)
        
        # 确保返回处理后的帧 
        with self.shared_state.frame_lock:  
            self.shared_state.processed_frame  = frame.copy()  
        
        return frame 
    
    def _handle_detection_mode(self, frame, tracking_state, current_time):
        """处理检测模式逻辑"""
        if current_time - tracking_state.last_detection_time  > tracking_state.detection_interval:  
            results = self.model(frame,  classes=[0], half=True, imgsz=320,
                                verbose=False, conf=0.4, iou=0.4)
            tracking_state.last_detection_time  = current_time 
            
            if results:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    cv2.rectangle(frame,  (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 处理手动指定的目标坐标 
            if tracking_state.target_coords:  
                self._init_manual_tracking(frame, tracking_state)
    
    def _init_manual_tracking(self, frame, tracking_state):
        """初始化手动跟踪"""
        click_x, click_y = tracking_state.target_coords    
        cv2.circle(frame,  (click_x, click_y), 10, (0,0,255), -1)
        
        # 创建虚拟检测框 (适当增大初始框大小)
        box_size = max(60, min(frame.shape[0],  frame.shape[1])  // 8)
        fake_box = [click_x-box_size//2, click_y-box_size//2, box_size, box_size]
        fake_detection = (fake_box, 0.99, 'person')
        
        # 初始化跟踪 
        tracks = tracking_state.tracker.update_tracks([fake_detection],  frame=frame)
        for track in tracks:
            if track.is_confirmed():   
                tracking_state.target_id  = track.track_id    
                tracking_state.is_tracking  = True 
                tracking_state.tracking_start_time  = time.time()   
                tracking_state.track_history.clear() 
                logger.info(f"Started  tracking manual target ID: {track.track_id}")  
                break 
        
        tracking_state.target_coords  = None 
    
    def _handle_manual_tracking(self, frame, tracking_state):
        """处理手动跟踪模式"""
        x, y = tracking_state.target_coords    
        
        # 绘制手动目标标记 
        cv2.circle(frame,  (x,y), 8, (0,0,255), -1)
        cv2.putText(frame,  "Manual Target", (x+10,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        # 创建虚拟检测框维持跟踪 (比初始化时稍大)
        box_size = max(80, min(frame.shape[0],  frame.shape[1])  // 6)
        fake_detection = ([x-box_size//2,y-box_size//2,box_size,box_size], 0.9, 'person')
        tracks = tracking_state.tracker.update_tracks([fake_detection],  frame=frame)
        
        tracking_state.target_coords  = None  # 单次使用 
        
        if tracks:
            self._update_tracking_data(frame, tracking_state, tracks)
            return True 
        return False 
    
    def _handle_automatic_tracking(self, frame, tracking_state, current_time, dt):
        """处理自动跟踪模式"""
        if current_time - tracking_state.last_detection_time  > tracking_state.tracking_interval:   
            results = self.model(frame,  classes=[0], half=True, imgsz=320,
                            verbose=False, conf=0.6, iou=0.4)
            tracking_state.last_detection_time  = current_time 
            
            if results and len(results[0].boxes) > 0:
                detections = []
                min_box_size = max(50, min(frame.shape[0],  frame.shape[1])  // 10)
                
                for box in results[0].boxes.cpu().numpy():   
                    x1,y1,x2,y2 = box.xyxy[0].astype(int)   
                    box_width, box_height = x2-x1, y2-y1 
                    
                    # 根据目标大小动态调整最小框大小要求 
                    if box_width > min_box_size and box_height > min_box_size:
                        score = box.conf[0] 
                        # 给靠近中心的检测框更高的分数 
                        center_dist = np.sqrt(((x1+x2)/2  - frame.shape[1]//2)**2  + 
                                            ((y1+y2)/2 - frame.shape[0]//2)**2) 
                        adjusted_score = score * (1 - min(center_dist/(frame.shape[1]//2),  0.5))
                        detections.append(([x1,y1,x2-x1,y2-y1],  adjusted_score, 'person'))
                
                if detections:
                    # 优先选择最接近上次位置的目标 
                    if tracking_state.last_target_center: 
                        detections.sort(key=lambda  d: np.sqrt( 
                            (d[0][0]+d[0][2]/2 - tracking_state.last_target_center[0])**2  +
                            (d[0][1]+d[0][3]/2 - tracking_state.last_target_center[1])**2  
                        ))
                    
                    tracks = tracking_state.tracker.update_tracks(detections,  frame=frame)
                    if tracks:
                        return self._update_tracking_data(frame, tracking_state, tracks, dt)
        return False 
    
    def _update_tracking_data(self, frame, tracking_state, tracks, dt=0.1):
        """更新跟踪数据并生成控制指令"""
        height, width = frame.shape[:2] 
        center_x, center_y = width // 2, height // 2 
        
        for track in tracks:
            if not track.is_confirmed(): 
                continue 
                
            # 如果已经设置了target_id，则只跟踪该ID 
            if tracking_state.target_id  and track.track_id  != tracking_state.target_id: 
                continue 
                
            tracking_state.target_id  = track.track_id    
            ltrb = track.to_ltrb().astype(int)   
            x1,y1,x2,y2 = ltrb 
            
            # 绘制跟踪框和信息 
            cv2.rectangle(frame,  (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame,  f"ID: {track.track_id}",  (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
            # 计算并平滑中心点 
            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            if not hasattr(tracking_state, 'smoothed_center') or tracking_state.smoothed_center  is None:
                tracking_state.smoothed_center  = current_center 
            else:
                # 使用动态平滑系数 (目标移动越快，平滑系数越小)
                motion = np.sqrt((current_center[0]  - tracking_state.smoothed_center[0])**2  +
                                (current_center[1] - tracking_state.smoothed_center[1])**2) 
                dynamic_factor = min(0.3, max(0.1, 0.3 - motion/100))
                tracking_state.smoothed_center  = (
                    int(dynamic_factor * current_center[0] + (1 - dynamic_factor) * tracking_state.smoothed_center[0]), 
                    int(dynamic_factor * current_center[1] + (1 - dynamic_factor) * tracking_state.smoothed_center[1]) 
                )
            
            # 更新跟踪历史 
            if not hasattr(tracking_state, 'track_history'):
                tracking_state.track_history  = collections.deque(maxlen=30) 
            tracking_state.track_history.append(tracking_state.smoothed_center) 
            tracking_state.last_target_center  = tracking_state.smoothed_center  
            
            # 绘制运动轨迹 (使用更平滑的曲线)
            if len(tracking_state.track_history)  > 1:
                points = np.array(tracking_state.track_history) 
                # 使用插值生成平滑曲线 
                if len(points) >= 4:
                    x = points[:, 0]
                    y = points[:, 1]
                    t = np.arange(len(points)) 
                    spl_x = make_interp_spline(t, x, k=3)
                    spl_y = make_interp_spline(t, y, k=3)
                    t_new = np.linspace(0,  len(points)-1, 50)
                    x_new = spl_x(t_new)
                    y_new = spl_y(t_new)
                    
                    for i in range(1, len(x_new)):
                        cv2.line(frame,  
                                (int(x_new[i-1]), int(y_new[i-1])),
                                (int(x_new[i]), int(y_new[i])),
                                (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    for i in range(1, len(points)):
                        cv2.line(frame,  points[i-1], points[i], (0, 255, 255), 2)
            
            # 在最新点上画一个小圆点 
            cv2.circle(frame,  tracking_state.smoothed_center,  5, (0, 0, 255), -1)
            
            # 生成控制指令 
            self._generate_control_commands(frame, tracking_state, dt)
            
            tracking_state.tracking_lost_time  = None 
            return True 
        
        return False 
    
    def _generate_control_commands(self, frame, tracking_state, dt):
        """生成并发送控制指令"""
        height, width = frame.shape[:2] 
        center_x, center_y = width // 2, height // 2 
        
        if not hasattr(tracking_state, 'smoothed_center') or tracking_state.smoothed_center  is None:
            return 
        
        # 计算误差（像素坐标）
        error_x = tracking_state.smoothed_center[0]  - center_x 
        error_y = tracking_state.smoothed_center[1]  - center_y 
        
        # 动态调整PID参数 (误差越大，P越大，I越小)
        self.yaw_pid.dynamic_adjust(abs(error_x)) 
        self.pitch_pid.dynamic_adjust(abs(error_y)) 
        
        # 获取控制输出 (考虑帧率影响)
        yaw_speed = self.yaw_pid.update(error_x,  dt)
        pitch_speed = self.pitch_pid.update(error_y,  dt)
        
        # 限制最大速度 (防止剧烈运动)
        max_speed = 0.5  # 根据机器人性能调整 
        yaw_speed = np.clip(yaw_speed,  -max_speed, max_speed)
        pitch_speed = np.clip(pitch_speed,  -max_speed, max_speed)
        
        # 绘制控制信息 
        cv2.putText(frame,  f"Yaw: {yaw_speed:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame,  f"Pitch: {pitch_speed:.2f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 发送控制命令 (限制控制频率)
        if time.time()  - tracking_state.last_control_time  > 0.05:  # 20Hz 
            # 将速度转换为机器人控制参数 
            yaw_param = int(yaw_speed * 32767)
            pitch_param = int(pitch_speed * 32767)
            self.commander.send_command(0x21010135,  yaw_param)
            self.commander.send_command(0x21010130,  pitch_param)
            tracking_state.last_control_time  = time.time() 
    
    def _handle_tracking_lost(self, tracking_state, current_time):
        """处理目标丢失情况"""
        if tracking_state.tracking_lost_time  is None:
            tracking_state.tracking_lost_time  = current_time 
            # 发送停止指令 
            self.commander.send_command(0x21010135,  0)
            self.commander.send_command(0x21010130,  0)
        elif current_time - tracking_state.tracking_lost_time  > 2.0:  # 2秒后重置 
            tracking_state.reset() 
            logger.info("Target  lost, switching to detection mode")
    
    def _update_mode_indicator(self, frame, width, height, tracking_state):
        """更新模式显示信息"""
        mode_text = "TRACKING" if tracking_state.is_tracking  else "DETECTION"
        color = (0, 255, 0) if tracking_state.is_tracking  else (0, 0, 255)
        
        cv2.putText(frame,  f"Mode: {mode_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 添加时间戳和状态信息 
        timestamp = time.strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame,  timestamp, (width-250, height-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 如果正在跟踪，显示跟踪持续时间 
        if tracking_state.is_tracking: 
            duration = int(time.time()  - tracking_state.tracking_start_time) 
            cv2.putText(frame,  f"Tracking: {duration}s", (width-250, height-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 
    def stop(self):
        """安全停止线程"""
        self.running  = False 
 
# RTSP推流线程（增强错误恢复）
class RTSPStreamThread(threading.Thread):
    def __init__(self, shared_state):
        super().__init__(name="RTSPStreamThread")
        self.shared_state  = shared_state 
        self.running  = True 
        self.ffmpeg_process  = None 
        self.retry_count  = 0 
        self.max_retries  = 10 
        self.frame_interval  = 0.033  # 默认30fps 
        self.last_valid_frame  = None 
        self.stream_active  = False 
        self.last_restart_time  = 0 
        self.frame_counter  = 0 
        self.last_log_time  = 0 
 
    def run(self):
        while self.running  and self.retry_count  < self.max_retries: 
            try:
                # 等待有效帧尺寸 
                while None in self.shared_state.frame_dimensions  and self.running: 
                    time.sleep(0.1) 
                
                if not self.running: 
                    break 
                
                # 初始化FFmpeg 
                if not self._init_ffmpeg():
                    continue 
                
                # 主推流循环 
                last_frame_time = time.time() 
                while self.running  and self.stream_active: 
                    try:
                        current_time = time.time() 
                        
                        # 控制帧率 
                        sleep_time = last_frame_time + self.frame_interval  - current_time 
                        if sleep_time > 0:
                            time.sleep(sleep_time) 
                        
                        last_frame_time = time.time() 
                        
                        # 获取帧 
                        frame = self._get_frame_with_timeout()
                        if frame is None:
                            continue 
                        
                        # 写入FFmpeg 
                        self._write_frame(frame)
                        
                        # 日志记录 
                        self.frame_counter  += 1 
                        if current_time - self.last_log_time  > 5:
                            fps = self.frame_counter  / (current_time - self.last_log_time) 
                            logger.info(f"Streaming  FPS: {fps:.1f}")
                            self.frame_counter  = 0 
                            self.last_log_time  = current_time 
                            
                    except BrokenPipeError:
                        logger.error("FFmpeg  pipe broken, restarting...")
                        self._restart_ffmpeg()
                        break 
                    except Exception as e:
                        logger.error(f"Stream  error: {e}")
                        self._restart_ffmpeg()
                        break 
                        
            except Exception as e:
                logger.error(f"Streaming  setup error: {e}")
                self._cleanup_ffmpeg()
                self.retry_count  += 1 
                time.sleep(min(5,  self.retry_count)) 
        
        if self.retry_count  >= self.max_retries: 
            logger.error("Max  retries reached for streaming")
            with self.shared_state.state_lock: 
                self.shared_state.stream_active  = False 
 
    def _init_ffmpeg(self):
        """初始化FFmpeg进程"""
        width, height = self.shared_state.frame_dimensions  
        fps = self.shared_state.fps  or 30 
        self.frame_interval  = 1.0 / fps 

        
        command = [
            'ffmpeg',
            '-re',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', '10',
            '-crf', '23',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            '-muxdelay', '0.1',
            Config.RTSP_OUTPUT_URL 
        ]
        
        logger.info(f"Starting  FFmpeg: {' '.join(command)}")
        try:
            self.ffmpeg_process  = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL 
            )
            
            # 检查进程是否启动成功 
            time.sleep(0.5) 
            if self.ffmpeg_process.poll()  is not None:
                error = self.ffmpeg_process.stderr.read().decode('utf-8',  errors='ignore') 
                raise RuntimeError(f"FFmpeg failed to start: {error[:200]}...")
            
            self.stream_active  = True 
            self.retry_count  = 0 
            return True 
            
        except Exception as e:
            logger.error(f"FFmpeg  init failed: {e}")
            self._cleanup_ffmpeg()
            return False 
 
    def _get_frame_with_timeout(self, timeout=0.1):
        """带超时的帧获取"""
        start_time = time.time() 
        while time.time()  - start_time < timeout:
            try:
                with self.shared_state.frame_lock: 
                    if self.shared_state.processed_frame  is not None:
                        return self.shared_state.processed_frame.copy() 
                    else:
                        logger.debug("Waiting  for processed frame...")
            except Exception as e:
                logger.warning(f"Frame  lock error: {e}")
            time.sleep(0.01) 
        
        if self.last_valid_frame  is not None:
            logger.debug("Using  last valid frame as fallback")
            return self.last_valid_frame  
            
        logger.warning("Frame  timeout reached, using blank frame")
        return np.zeros((480,  640, 3), dtype=np.uint8) 
 
    def _write_frame(self, frame):
        """写入帧到FFmpeg"""
        try:
            self.ffmpeg_process.stdin.write(frame.tobytes()) 
            self.ffmpeg_process.stdin.flush() 
            self.last_valid_frame  = frame.copy() 
        except Exception as e:
            raise BrokenPipeError(f"FFmpeg write failed: {e}")
 
    def _restart_ffmpeg(self):
        """安全重启FFmpeg"""
        now = time.time() 
        if now - self.last_restart_time  < 5:  # 5秒内不重复重启 
            time.sleep(5  - (now - self.last_restart_time)) 
        
        self._cleanup_ffmpeg()
        self.stream_active  = False 
        self.last_restart_time  = time.time() 
 
    def _cleanup_ffmpeg(self):
        """清理FFmpeg资源"""
        if self.ffmpeg_process: 
            try:
                if self.ffmpeg_process.stdin: 
                    try:
                        self.ffmpeg_process.stdin.close() 
                    except:
                        pass 
                
                self.ffmpeg_process.terminate() 
                self.ffmpeg_process.wait(timeout=1) 
            except:
                pass 
            finally:
                self.ffmpeg_process  = None 
 
    def stop(self):
        """安全停止线程"""
        self.running  = False 
        self._cleanup_ffmpeg()
 
# 控制线程（优化队列处理）
class ControlThread(threading.Thread):
    def __init__(self, shared_state):
        super().__init__(name="ControlThread")
        self.shared_state  = shared_state 
        self.running  = True 
        self.commander  = RobotCommander()
        self.last_command_time  = 0 
        self.command_interval  = 0.05  # 20fps 
 
    def run(self):
        while self.running  and self.shared_state.stream_active: 
            try:
                # 控制命令发送频率 
                current_time = time.time() 
                if current_time - self.last_command_time  < self.command_interval: 
                    time.sleep(0.001) 
                    continue 
                
                # 从队列获取命令（非阻塞）
                try:
                    code, param = self.shared_state.control_queue.get_nowait() 
                    self.commander.send_command(code,  param)
                    self.last_command_time  = time.time() 
                    logger.debug(f"Sent  control command: {code}, {param}")
                except queue.Empty:
                    time.sleep(0.01) 
                
            except Exception as e:
                logger.error(f"Control  thread error: {e}")
                time.sleep(0.1) 
 
    def stop(self):
        """安全停止线程"""
        self.running  = False 
 
# 主函数（增强资源监控）
def main():
    global shared_state 
    
    # 初始化共享状态 
    shared_state = SharedState()
    
    # 资源监控线程 
    def monitor_resources():
        while getattr(shared_state, 'stream_active', True):
            try:
                time.sleep(5) 
                logger.info(f"[Monitor]  Threads: {threading.active_count()}") 
                logger.info(f"[Monitor]  Frame queue: {shared_state.frame_queue.qsize()}/{Config.MAX_FRAME_QUEUE_SIZE}") 
                logger.info(f"[Monitor]  Control queue: {shared_state.control_queue.qsize()}/{Config.MAX_CONTROL_QUEUE_SIZE}") 
            except:
                pass 
    
    monitor_thread = threading.Thread(target=monitor_resources, name="MonitorThread", daemon=True)
    monitor_thread.start() 
    
    # 初始化YOLO模型 
    try:
        logger.info("Initializing  YOLO model...")
        model = YOLO('yolov8n.pt').to(device)  
        
        # 模型预热 
        logger.info("Warming  up model...")
        _ = model(np.zeros((320,  320, 3), dtype=np.uint8),  verbose=False)
        logger.info("Model  ready")
    except Exception as e:
        logger.error(f"Failed  to initialize model: {e}")
        return 
    
    # 初始化PID控制器 
    yaw_pid = EnhancedPIDController(kp=0.003, ki=0.0002, kd=0.0008, max_speed=1.0, dead_zone=8)
    pitch_pid = EnhancedPIDController(kp=0.0015, ki=0.0002, kd=0.0008, max_speed=1.0, dead_zone=8)
    
    # 启动HTTP服务器线程 
    http_thread = threading.Thread(target=start_http_server, name="HTTPServerThread", daemon=True)
    http_thread.start()  
    
    # 启动视频捕获线程 
    video_thread = VideoCaptureThread(shared_state)
    video_thread.start()  
    
    # 等待获取第一帧以确定视频尺寸 
    logger.info("Waiting  for first frame...")
    start_time = time.time() 
    while None in shared_state.frame_dimensions  and shared_state.stream_active  and (time.time()  - start_time < 15):
        time.sleep(0.1) 
    
    if None in shared_state.frame_dimensions: 
        logger.error("Failed  to initialize video stream")
        video_thread.stop() 
        return 
    
    logger.info(f"Video  initialized: {shared_state.frame_dimensions}") 
    
    width, height = shared_state.frame_dimensions  
    logger.info(f"Video  initialized successfully:")
    logger.info(f"   Resolution: {width}x{height}")
    logger.info(f"   FPS: {shared_state.fps:.1f}") 
    logger.info(f"   Codec: {getattr(shared_state, 'codec', 'unknown')}")
     
    # 启动检测/跟踪线程 
    detection_thread = DetectionTrackingThread(shared_state, model, yaw_pid, pitch_pid)
    detection_thread.start()  
    
    # 启动RTSP推流线程 
    rtsp_thread = RTSPStreamThread(shared_state)
    rtsp_thread.start()  
    
    # 启动控制线程 
    control_thread = ControlThread(shared_state)
    control_thread.start()  
    
    try:
        # 主循环 
        while shared_state.stream_active: 
            time.sleep(1) 
            
    except KeyboardInterrupt:
        logger.info("Received  keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected  error: {e}")
    finally:
        # 清理资源 
        logger.info("Starting  shutdown sequence...")
        
        with shared_state.state_lock: 
            shared_state.stream_active  = False 
            shared_state.detection_active  = False 
        
        # 停止线程 
        for thread in [video_thread, detection_thread, rtsp_thread, control_thread]:
            if thread.is_alive(): 
                thread.stop() 
                thread.join(timeout=2) 
                if thread.is_alive(): 
                    logger.warning(f"Thread  {thread.name}  did not stop gracefully")
        
        logger.info("Application  shutdown complete")
 
if __name__ == "__main__":
    main()