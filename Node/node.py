import cv2
import io
import socket
import struct
import time
import pickle
import zlib

import numpy as np
import math

import queue
import threading
from threading import Thread

############## 전역 변수 ##############
# 노드, 에지가 한 프레임을 처리하는데 걸리는 추정 시간
node_time = 0.00
edge_time = 0.005
# 지연 시간을 계산할 때 사용할 상수
a = 0.6

############## 이미지 처리 변수 ##############
# 이미지 사이즈(cols, rows)를 초기화
width = 1280
height = 720

w = width
h = height
K = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]])

# 처리할 프레임 개수
img_cnt = 0

# q_max size
q_max = 50000
# 해당 순서의 프레임을 노드와 에지중 어느곳에서 처리 했는지 표시
procSeq = queue.Queue(q_max)
procSeq2 = queue.Queue(q_max)
# 노드, 에지에서 영상처리를 해야하는 Mat 저장
nodeBeforeBuff = queue.Queue(q_max)
edgeBeforeBuff = queue.Queue(q_max)
# 노드, 에지에서 영상처리를 마친 Mat 저장
nodeAfterBuff = queue.Queue(q_max)
edgeAfterBuff = queue.Queue(q_max)

nodebeforebufftime = queue.Queue(q_max)
edgebeforebufftime = queue.Queue(q_max)

nodeafterbufftime = queue.Queue(q_max)
edgeafterbufftime = queue.Queue(q_max)

####### 통신 변수 #######
# ip addr
ip_konkuk = '114.70.22.26'
ip_home = '127.0.0.1'
ip_phone = '172.20.10.2'
ip = ip_konkuk
clnt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clnt_sock.connect((ip, 8485))
connection = clnt_sock.makefile('wb')
# 통신 끝을 알리는 메세지
msg_stop = 'stop'

####### opencv video 변수 #######
cam = cv2.VideoCapture('I2.mp4')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

f = open("C:/Users/강지완/Desktop/Node/test4.txt", "w")
############## ImageProcess 함수 ##############



### convert_pt END

### img_mod : frame을 VR영상으로 변환
def img_mod():
    while True:
        frame = nodeBeforeBuff.get()
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        h_, w_ = frame.shape[:2]
        # pixel coordinates
        y_i, x_i = np.indices((h_, w_))
        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T  # normalized coords
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
        B = K.dot(A.T).T  # project back to image-pixels plane
        # back from homog coords
        B = B[:, :-1] / B[:, [-1]]
        # make sure warp coords only within image bounds
        B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
        B = B.reshape(h_, w_, -1)

        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # for transparent borders...
        # warp the image according to cylindrical coords
        frame = cv2.remap(frame_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA,
                          borderMode=cv2.BORDER_TRANSPARENT)
        nodeafterbufftime.put(time.time())
        # if node_time != 0.0:
        #     node_time = node_time * a + cur_node_time * (1 - a);
        # else:
        #     node_time = cur_node_time
        nodeAfterBuff.put(frame)
        # cv2.imshow('ImageWindow', frame)
        # cv2.waitKey(1)


### img_mod END

############## Socket Communicate 함수 ##############
### recv_img : edge로부터 처리한 영상을 받는다.
def recv_img():
    data = b""
    # clnt_sock.send(msg_go.encode('utf-8'))
    # calcsize @시스템에 따름. = 시스템에 따름 < 리틀 엔디안 > 빅 엔디안 !네트워크(빅 엔디안)
    # 원본 >L
    payload_size = struct.calcsize(">L")
    #print("payload_size: {}".format(payload_size))

    while len(data) < payload_size:
        #print("Recv: {}".format(len(data)))
        data += clnt_sock.recv(4096)

    #print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    #print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += clnt_sock.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    # cv2.imshow('ImageWindow', frame)
    # cv2.waitKey(1)
    edgeafterbufftime.put(time.time())
    edgeAfterBuff.put(frame)


# frames.append(frame)

# cv2.imshow('ImageWindow', frame)
# cv2.waitKey(40)
### recv_img END

### send_img :
def send_img():
    frame = edgeBeforeBuff.get()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)

    #print("{}: {}".format(img_cnt, size))
    clnt_sock.sendall(struct.pack(">L", size) + data)


### send_img END

### sock_commu() :
def sock_commu():
    global edge_time
    while True:
        start_time = time.time()
        send_img()
        recv_img()
        cur_edge_time = time.time()

        # if edge_time != 0.0:
        #     edge_time = edge_time * a + cur_edge_time * (1 - a)
        # else:
        #     edge_time = cur_edge_time


### sock_commu() END

############## Judge Algorithm 함수 ##############
def judge_algo():
    global img_cnt
    global node_time
    global edge_time

    while (cam.isOpened()):
        ret, frame = cam.read()

        if frame is None:
            break

        # result, frame = cv2.imencode('.jpg', frame, encode_param)
        if nodeafterbufftime.qsize() > 0:
            recentonenode = nodeafterbufftime.get() - nodebeforebufftime.get()
            #f.write(str(recentonenode) + '\n')
            #print("node: "+str(recentonenode))
            node_time = (1-a) * node_time + a * (recentonenode)
            #print(node_time)
        node_latency = node_time * (nodeBeforeBuff.qsize()+1)
        if nodeBeforeBuff.qsize()==0:
            node_latency=0
        if edgebeforebufftime.qsize() > 0:
            recentoneedge = edgeafterbufftime.get() - edgebeforebufftime.get()
            #f.write(str(recentoneedge) + '\n')
            #print("edge: "+str(recentoneedge))
            edge_time = (1-a) * edge_time + a * (recentoneedge)
           # print(edge_time)
        edge_latency = edge_time * (edgeBeforeBuff.qsize()+1)
        if edgeBeforeBuff.qsize()==0:
            edge_latency=0
        if node_latency <= edge_latency:
            #f.write(str(node_latency) + '\n')
            procSeq.put(1)
            procSeq2.put(1)
            nodebeforebufftime.put(time.time())
            nodeBeforeBuff.put(frame)
        else:
            #f.write(str(edge_latency) + '\n')
            procSeq.put(2)
            procSeq2.put(2)
            edgebeforebufftime.put(time.time())
            edgeBeforeBuff.put(frame)

        img_cnt += 1
    procSeq.put(0)
    cam.release()


############## Img Merger 함수 ##############
def img_merger():
    while True:
        serv_num = procSeq.get()
        if serv_num == 1:
            frame = nodeAfterBuff.get()
            print("hi")

            # cv2.imshow('ImageWindow', frame)
            # cv2.waitKey(40)
        elif serv_num == 2:
            frame = edgeAfterBuff.get()

            # cv2.imshow('ImageWindow', frame)
            # cv2.waitKey(40)
        else:
            break


### threads_func :
if __name__ == '__main__':
    thread_img_process = threading.Thread(target=img_mod, daemon=True)
    thread_sock_commu = threading.Thread(target=sock_commu, daemon=True)
    thread_judge_algo = threading.Thread(target=judge_algo, daemon=True)
    # thread_img_merger = threading.Thread(target = img_merger)
    thread_img_process.start()
    thread_sock_commu.start()
    thread_judge_algo.start()
    # thread_img_merger.start()
    # thread_img_merger.join()
    while True:
        start=time.time()
        serv_num = procSeq.get()
        if serv_num == 1:
            #get 함수는 비어있으면 들어올 때 까지 기다린다.
            frame = nodeAfterBuff.get()

        elif serv_num == 2:
            frame = edgeAfterBuff.get()
        else:
            break
        cv2.imshow('ImageWindow', frame)
        end=time.time()-start
        print(end)
        f.write(str(end)+'\n')
        cv2.waitKey(1)
    f.close()
    clnt_sock.send(msg_stop.encode('utf-8'))
    clnt_sock.close()
#f.close()
### threads_func END