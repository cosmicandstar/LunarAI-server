import base64
import cv2
import zmq
import time

context = zmq.Context()

camera = cv2.VideoCapture('tcp://192.168.0.26:3003')  # 카메라와 연결

send_socket = context.socket(zmq.PUB)
send_socket.bind('tcp://*:5000')
last_time = time.perf_counter()

while camera.isOpened():
    grabbed, frame = camera.read()
    if (time.perf_counter() - last_time) > (1 / 10.0):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        encoded, buffer = cv2.imencode('.jpg', gray_frame)
        jpg_as_text = base64.b64encode(buffer)
        send_socket.send(jpg_as_text)
        last_time = time.perf_counter()