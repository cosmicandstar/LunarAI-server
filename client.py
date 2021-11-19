import base64
import cv2
import zmq
import time

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.bind('tcp://*:4000')

camera = cv2.VideoCapture(0)  # 카메라와 연결

while True:
    old_time = time.perf_counter()
    grabbed, frame = camera.read()
    #frame = cv2.resize(frame, (640, 480))  # 사이즈 조절
    encoded, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    footage_socket.send(jpg_as_text)
    while (time.perf_counter() - old_time) <= (1 / 10.0):
        pass