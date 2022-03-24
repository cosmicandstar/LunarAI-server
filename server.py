import base64
import cv2
import zmq
import numpy as np
import threading


def face_model_fit(id):
    # todo 얼굴 학습되게 코드 넣기
    print("fit :", id)


def process(recv_socket):
    while True:
        frame = recv_socket.recv_string()
        data = frame.split(":")
        header = data[0]
        if header == "fit":
            id = data[1]
            face_model_fit(id)


def face_model(recv_socket, send_socket):
    while True:
        frame = recv_socket.recv_string()
        img = base64.b64decode(frame)
        np_img = np.frombuffer(img, dtype=np.uint8)
        source = cv2.imdecode(np_img, 1)

        # todo 모델 코드 작성 (현재는 흑백으로 변환되게 해둠)
        gray_frame = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        encoded, buffer = cv2.imencode(".jpg", gray_frame)
        jpg_as_text = base64.b64encode(buffer)
        send_socket.send(jpg_as_text)


context = zmq.Context()

send_socket = context.socket(zmq.PUB)
send_socket.bind("tcp://*:5000")

recv_socket = context.socket(zmq.SUB)
recv_socket.connect("tcp://192.168.0.14:4000")  # 라즈베리파이 ip
recv_socket.setsockopt_string(zmq.SUBSCRIBE, np.compat.unicode(""))

recv_socket2 = context.socket(zmq.SUB)
recv_socket2.connect("tcp://localhost:6000")
recv_socket2.setsockopt_string(zmq.SUBSCRIBE, np.compat.unicode(""))

face_model_thread = threading.Thread(target=face_model, args=(recv_socket, send_socket))
process_thread = threading.Thread(target=process, args=(recv_socket2,))

face_model_thread.start()
process_thread.start()
