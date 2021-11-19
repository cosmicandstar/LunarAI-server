import base64
import cv2
import zmq
import numpy as np

context = zmq.Context()

recv_socket = context.socket(zmq.SUB)
recv_socket.connect('tcp://localhost:4000')  # 라즈베리파이 ip
recv_socket.setsockopt_string(zmq.SUBSCRIBE, np.compat.unicode(''))

send_socket = context.socket(zmq.PUB)
send_socket.bind('tcp://*:5000')

while True:
    frame = recv_socket.recv_string()
    img = base64.b64decode(frame)
    np_img = np.frombuffer(img, dtype=np.uint8)
    source = cv2.imdecode(np_img, 1)
    gray_frame = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    encoded, buffer = cv2.imencode('.jpg', gray_frame)
    jpg_as_text = base64.b64encode(buffer)
    send_socket.send(jpg_as_text)