"""
Unity通信模块 - Python端
负责向Unity推送视频流数据，接收Unity的pose切换命令
"""
import socket
import json
import threading
import time
import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import struct


class UnityCommunication:
    """
    Unity通信管理器
    使用TCP socket进行双向通信
    """
    
    def __init__(self, host: str = "127.0.0.1", video_port: int = 8888, control_port: int = 8889):
        """
        初始化Unity通信
        
        Args:
            host: Unity服务器地址
            video_port: 视频流推送端口
            control_port: 控制命令接收端口
        """
        self.host = host
        self.video_port = video_port
        self.control_port = control_port
        
        self.video_socket: Optional[socket.socket] = None
        self.video_server: Optional[socket.socket] = None
        self.control_socket: Optional[socket.socket] = None
        self.control_server: Optional[socket.socket] = None
        self.control_client: Optional[socket.socket] = None
        
        self.is_connected = False
        self.is_listening = False
        
        # 回调函数
        self.on_pose_switch: Optional[callable] = None
        
        # 数据格式：帧头 + 数据长度 + JSON元数据 + 图像数据
        # 帧头：4字节 "FRAM"
        # 数据长度：4字节 uint32
        # JSON元数据长度：4字节 uint32
        # JSON元数据：UTF-8字符串
        # 图像数据：BGR格式的字节数组
    
    def connect(self) -> bool:
        """
        连接到Unity
        
        Python作为服务端，Unity作为客户端
        - 视频流端口8888: Python监听，Unity连接
        - 控制端口8889: Python监听，Unity连接发送命令
        
        Returns:
            是否连接成功
        """
        try:
            # 启动视频流推送服务器（等待Unity连接）
            self.video_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.video_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.video_server.bind((self.host, self.video_port))
            self.video_server.listen(1)
            print(f"[Unity通信] 视频流服务器已启动，等待Unity连接... {self.host}:{self.video_port}")
            
            # 启动视频连接等待线程（非阻塞）
            self.is_listening = True
            video_accept_thread = threading.Thread(target=self._accept_video_client, daemon=True)
            video_accept_thread.start()
            
            # 启动控制命令接收服务器（等待Unity连接发送命令）
            self.control_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.control_server.bind((self.host, self.control_port))
            self.control_server.listen(1)
            print(f"[Unity通信] 控制服务器监听在 {self.host}:{self.control_port}")
            
            # 启动监听线程
            listen_thread = threading.Thread(target=self._listen_control, daemon=True)
            listen_thread.start()
            
            self.is_connected = False  # 初始未连接，等待Unity连接后设置为True
            return True
            
        except Exception as e:
            print(f"[Unity通信] 启动服务器失败: {e}")
            self.is_connected = False
            return False
    
    def _accept_video_client(self):
        """等待Unity客户端连接到视频流端口"""
        while self.is_listening:
            try:
                if self.video_server:
                    client_socket, addr = self.video_server.accept()
                    print(f"[Unity通信] Unity客户端已连接到视频流端口: {addr}")
                    self.video_socket = client_socket
                    self.is_connected = True
                    break  # 只接受一个客户端
            except Exception as e:
                if self.is_listening:
                    print(f"[Unity通信] 接受视频客户端连接错误: {e}")
                time.sleep(0.1)
    
    def disconnect(self):
        """断开连接"""
        self.is_connected = False
        self.is_listening = False
        
        if self.video_socket:
            try:
                self.video_socket.close()
            except:
                pass
            self.video_socket = None
        
        if self.video_server:
            try:
                self.video_server.close()
            except:
                pass
            self.video_server = None
        
        if self.control_client:
            try:
                self.control_client.close()
            except:
                pass
            self.control_client = None
        
        if self.control_server:
            try:
                self.control_server.close()
            except:
                pass
            self.control_server = None
        
        print("[Unity通信] 已断开连接")
    
    def _listen_control(self):
        """监听Unity的控制命令"""
        while self.is_listening:
            try:
                if self.control_server:
                    self.control_client, addr = self.control_server.accept()
                    print(f"[Unity通信] Unity客户端已连接: {addr}")
                    
                    while self.is_listening:
                        # 接收命令长度
                        length_data = self.control_client.recv(4)
                        if len(length_data) < 4:
                            break
                        
                        length = struct.unpack('>I', length_data)[0]
                        
                        # 接收JSON命令
                        command_data = b''
                        while len(command_data) < length:
                            chunk = self.control_client.recv(length - len(command_data))
                            if not chunk:
                                break
                            command_data += chunk
                        
                        if len(command_data) == length:
                            try:
                                command = json.loads(command_data.decode('utf-8'))
                                self._handle_command(command)
                            except Exception as e:
                                print(f"[Unity通信] 处理命令失败: {e}")
                    
                    if self.control_client:
                        self.control_client.close()
                        self.control_client = None
                        print("[Unity通信] Unity客户端已断开")
                        
            except Exception as e:
                if self.is_listening:
                    print(f"[Unity通信] 监听错误: {e}")
                time.sleep(0.1)
    
    def _handle_command(self, command: Dict):
        """处理Unity发送的命令"""
        cmd_type = command.get('type')
        
        if cmd_type == 'switch_pose':
            pose_index = command.get('pose_index', 0)
            direction = command.get('direction')  # 'next' or 'previous'
            
            print(f"[Unity通信] 收到pose切换命令: {command}")
            
            if self.on_pose_switch:
                self.on_pose_switch(pose_index, direction)
        else:
            print(f"[Unity通信] 未知命令类型: {cmd_type}")
    
    def send_frame(self, camera_id: int, frame: np.ndarray, match_score: float, 
                   pose_name: str = "") -> bool:
        """
        发送一帧数据到Unity
        
        Args:
            camera_id: 摄像头ID (0 或 1)
            frame: BGR格式的图像帧 (numpy array)
            match_score: 姿态匹配分数 (0-100)
            pose_name: 当前目标姿态名称
            
        Returns:
            是否发送成功
        """
        if not self.is_connected or not self.video_socket:
            return False
        
        try:
            # 编码图像
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, img_encoded = cv2.imencode('.jpg', frame, encode_param)
            
            if not result:
                return False
            
            # 构建元数据
            metadata = {
                'camera_id': camera_id,
                'timestamp': time.time(),
                'match_score': round(match_score, 2),
                'pose_name': pose_name,
                'width': frame.shape[1],
                'height': frame.shape[0],
                'format': 'jpg'
            }
            
            metadata_json = json.dumps(metadata).encode('utf-8')
            
            # 构建数据包
            # 帧头 (4字节) + 元数据长度 (4字节) + 图像长度 (4字节) + 元数据 + 图像数据
            frame_header = b'FRAM'
            metadata_len = struct.pack('>I', len(metadata_json))
            image_len = struct.pack('>I', len(img_encoded))
            
            packet = frame_header + metadata_len + image_len + metadata_json + img_encoded.tobytes()
            
            # 发送数据包
            self.video_socket.sendall(packet)
            
            return True
            
        except Exception as e:
            print(f"[Unity通信] 发送帧失败: {e}")
            self.is_connected = False
            return False
    
    def send_dual_frames(self, frame0: np.ndarray, score0: float,
                         frame1: np.ndarray, score1: float,
                         pose_name: str = "") -> bool:
        """
        同时发送两个摄像头的帧
        
        Args:
            frame0: 摄像头0的图像帧
            frame1: 摄像头1的图像帧
            score0: 摄像头0的匹配分数
            score1: 摄像头1的匹配分数
            pose_name: 当前目标姿态名称
            
        Returns:
            是否发送成功
        """
        success0 = self.send_frame(0, frame0, score0, pose_name)
        success1 = self.send_frame(1, frame1, score1, pose_name)
        return success0 and success1

