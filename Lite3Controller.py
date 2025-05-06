import socket
import struct
import tkinter as tk
import threading
import time

class RobotCommander:
    def __init__(self, local_port=20001, ctrl_ip="192.168.1.120", ctrl_port=43893, angle_port=54321):
        self.local_port = local_port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.ctrl_addr = (ctrl_ip, ctrl_port)
        
        # 设置角度接收socket
        self.angle_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.angle_socket.bind(('0.0.0.0', angle_port))
        
        self.root = tk.Tk()
        self.root.title("lite3控制接口")
        self.current_key_label = tk.Label(self.root, text="当前按键：")
        self.current_angle_label = tk.Label(self.root, text="当前角度(rad): 0.0")
        self.create_buttons()
        self.bind_keyboard()
        
        # 启动角度接收线程
        self.angle_thread = threading.Thread(target=self.receive_angles, daemon=True)
        self.angle_thread.start()
        
        # 当前角度值
        self.current_angle = 0.0

    def create_buttons(self):
        button_data = [
            ("基本状态", [
                ("心跳(k)", 0x21040001),
                ("回零(X)", 0x21010C05),
                ("起立/趴下(Z)", 0x21010202),
                ("原地模式", 0x21010D05),
                ("移动模式(C)", 0x21010D06),
            ]),
            ("步态", [
                ("低速", 0x21010300),
                ("中速(R)", 0x21010307),
                ("高速(T)", 0x21010303),
                ("正常/匍匐(Y)", 0x21010406),
                ("抓地(U)", 0x21010402),
                ("越障(I)", 0x21010401),
                ("高踏步(V)", 0x21010407),

            ]),
            ("动作", [
                ("扭身体", 0x21010204),
                ("翻身", 0x21010205),
                ("太空步", 0x2101030C),
                ("后空翻", 0x21010502),
                ("打招呼", 0x21010507),
                ("向前跳", 0x2101050B),
                ("扭身跳", 0x2101020D),
            ]),
            ("移动", [
                ("前进(W)", 0x21010130, 15000,0),
                ("后退(S)", 0x21010130, -15000,0),
                ("左平移(A)", 0x21010131, -30000,0),
                ("右平移(D)", 0x21010131, 30000,0),
                ("左转(Q)", 0x21010135, -32000,0),
                ("右转(E)", 0x21010135, 32000,0),
            ]),
            ("模式切换", [
                ("手动模式(B)", 0x21010C02),
                ("导航模式(N)", 0x21010C03),
                ("软急停(P)", 0x21010C0E),
                ("保存数据！！！", 0x21010C01)
            ])
        ]

        button_font = ("黑体", 11)
        button_bg_color = "#FFFFFF"
        button_active_bg_color = "#FFFACD"

        for group_row, (group_name, group_buttons) in enumerate(button_data):
            group_frame = tk.LabelFrame(self.root, text=group_name, font=button_font)
            group_frame.grid(row=group_row, column=0, padx=10, pady=4, sticky="nsew")

            for button_col, (text, command, *params) in enumerate(group_buttons):
                button = tk.Button(
                    group_frame,
                    text=text,
                    font=button_font,
                    bg=button_bg_color,
                    activebackground=button_active_bg_color,
                    command=lambda cmd=command, p=params: self.send_command(cmd, *p)
                )
                button.grid(row=0, column=button_col, pady=5, padx=4, sticky="nsew")
                button.config(fg='black')
        for col in range(len(button_data[0][1])):
            self.root.grid_columnconfigure(col, weight=1)

        for row in range(len(button_data)):
            self.root.grid_rowconfigure(row, weight=1)

        self.current_key_label.grid(row=len(button_data), column=0, pady=10)
        self.current_key_label.grid(row=len(button_data), column=0, pady=10)
        self.current_angle_label.grid(row=len(button_data)+1, column=0, pady=10)

    def bind_keyboard(self):
        self.root.bind("<KeyPress>", self.handle_key_press)
        self.root.bind("<KeyRelease>", self.handle_key_release)

    def handle_key_press(self, event):
        key_to_command = {
            'z': (0x21010202, 0),        # 起立/趴下
            'x': (0x21010C05, 0),        # 原地模式
            'c': (0x21010D06, 0),        # 移动模式
            'w': (0x21010130, 32767),    # 前进
            's': (0x21010130, -32767),   # 后退
            'a': (0x21010131, -32767),   # 左平移
            'd': (0x21010131, 32767),    # 右平移
            'q': (0x21010135, -32767),   # 左转
            'e': (0x21010135, 32767),    # 右转
            'r': (0x21010307, 0),        # 中速
            't': (0x21010303, 0),        # 高速
            'y': (0x21010406, 0),        # 正常/匍匐
            'u': (0x21010402, 0),        # 抓地
            'i': (0x21010401, 0),        # 越障
            'v': (0x21010407, 0),        # 高踏步
            'b': (0x21010C02, 0),        # 手动
            'n': (0x21010C03, 0),        # 导航
            'p': (0x21010C0E, 0),        # 软急停
        }
        key = event.char.lower()
        if key in key_to_command:
            code, param = key_to_command[key]
            self.send_command(code, param)
            self.update_current_key_label(key)

    def handle_key_release(self, event):
        key_to_command = {
            'w': (0x21010130, 0),      # 停止前进
            's': (0x21010130, 0),      # 停止后退
            'a': (0x21010131, 0),      # 停止左平移
            'd': (0x21010131, 0),      # 停止右平移
            'q': (0x21010135, 0),      # 停止左转
            'e': (0x21010135, 0),      # 停止右转
        }
        key = event.char.lower()
        if key in key_to_command:
            code, param = key_to_command[key]
            self.send_command(code, param)
            self.update_current_key_label(key)

    def receive_angles(self):
        while True:
            try:
                data, addr = self.angle_socket.recvfrom(1024)
                radians = float(data.decode('utf-8'))
                self.current_angle = radians
                
                # 更新UI显示
                self.root.after(0, self.update_angle_label, radians)
                
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

    def update_angle_label(self, angle):
        self.current_angle_label.config(text=f"当前角度(rad): {angle:.4f}")

    def continuous_command(self, code, param1=0, param2=0):
        self.send_simple(code, param1, param2)
        self.after_id = self.root.after(100, self.continuous_command, code, param1, param2)

    def send_command(self, code, param1=0, param2=0):
        print(f"发送命令：Code={code}, Param1={param1}, Param2={param2}")
        self.send_simple(code, param1, param2)

    def send_simple(self, code, param1=0, param2=0):
        try:
            payload = struct.pack('<3i', code, param1, param2)
            self.server.sendto(payload, self.ctrl_addr)
        except Exception as e:
            print(f"发送命令时出错：{e}")

    def update_current_key_label(self, key):
        self.current_key_label.config(text=f"当前按键：{key.upper()}")

    def start_heartbeat(self):
        self.continuous_command(0x21040001)

    def stop_continuous_command(self):
        if hasattr(self, 'after_id'):
            self.root.after_cancel(self.after_id)

    def on_closing(self):
        self.stop_continuous_command()
        self.server.close()
        self.angle_socket.close()
        self.root.destroy()

    def run(self):
        self.start_heartbeat()  # 在运行 GUI 之前开始发送心跳指令
        self.root.mainloop()

if __name__ == "__main__":
    gui = RobotCommander()
    gui.run()