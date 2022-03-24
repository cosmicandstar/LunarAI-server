import os
import facemodel.face
import datetime

# comment out below line to enable tensorflow logging outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pymysql

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import zmq
import base64
import threading
import boto3
import requests
from os import path
import json

# face_detector = cv2.CascadeClassifier(
#     "./opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
# )  # 경로 주의
# recognizer = cv2.face.LBPHFaceRecognizer_create()


def sendMessage(url):

    # friends list Get
    access_token = "UYJaPnQmzc6YSA_6OtdZWy37vajMqpblAteijworDR4AAAF9fT6krQ"
    print(access_token)
    kakaofriends_url = "https://kapi.kakao.com/v1/api/talk/friends"
    header = {"Authorization": "Bearer " + access_token}
    result = json.loads(requests.get(kakaofriends_url, headers=header).text)["elements"]
    friends_id = []
    for friend in result:
        friends_id.append(str(friend.get("uuid")))

    # message Send
    kakaotalk_url = "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"
    uuidsData = {"receiver_uuids": json.dumps(friends_id)}
    header = {"Authorization": "Bearer " + access_token}
    testObject = {
        "object_type": "feed",
        "content": {
            "title": "침입자 감지!",
            "description": "외부인이 [동아리방]에 침입하였습니다.",
            "image_url": url,
            "image_width": 640,
            "image_height": 640,
            "link": {
                "web_url": "http://www.daum.net",
                "mobile_web_url": "http://m.daum.net",
                "android_execution_params": "contentId=100",
                "ios_execution_params": "contentId=100",
            },
        },
        "buttons": [
            {
                "title": "웹으로 이동",
                "link": {"web_url": "https://naver.com"},
            }
        ],
    }
    data = {"template_object": json.dumps(testObject)}
    uuidsData.update(data)
    result = requests.post(kakaotalk_url, headers=header, data=uuidsData).status_code


def save_log(img_url, status, person_name):
    con = pymysql.connect(
        host="localhost",
        user="root",
        port=3307,
        password="123123",
        db="lunarai",
        charset="utf8",
    )  # 데이터베이스 접속 부분
    curs = con.cursor()
    sql = (
        "INSERT INTO lunarai.cctvlog(log_img, status, person_name) VALUES('"
        + img_url
        + "', '"
        + status
        + "', '"
        + person_name
        + "');"
    )
    curs.execute(sql)
    con.commit()

    con.close()


def face_model_fit(face_id):
    print("start fit")
    global recognizer
    face_dataset = []
    image_path = f"C:/hs/Project/LunarAI-web/streaming/{face_id}"
    imagePaths = [f"{image_path}/{f}" for f in os.listdir(image_path)]
    face_detector = cv2.CascadeClassifier(
        "./opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
    )  # 경로 주의
    for imagePath in imagePaths:
        img = Image.open(imagePath).convert("L")
        img_numpy = np.array(img, "uint8")
        faces = face_detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            # Save the captured image into the datasets folder
            # User 이름 폴더 안에 개수만큼 저장

            face_dataset.append(img_numpy[y : y + h, x : x + w])
    recognizer_new = cv2.face.LBPHFaceRecognizer_create()
    recognizer_new.train(face_dataset, np.array([face_id] * len(face_dataset)))

    recognizer = recognizer_new

    # Save the model into trainer/trainer.yml
    recognizer.write(
        "trainer/trainer.yml"
    )  # recognizer.save() worked on Mac, but not on Pi
    # Print the numer of facess trained and end program
    print("\n [INFO] {0} faces trained.".format(len(np.unique(face_id))))

    # todo 얼굴 학습되게 코드 넣기
    print("fit :", int(face_id))


def process(recv_socket):
    while True:
        frame = recv_socket.recv_string()
        data = frame.split(":")
        header = data[0]
        if header == "fit":
            id = data[1]
            face_model_fit(int(id))


def face_model(recv_socket, send_socket):
    global recognizer

    flag_weights = "./checkpoints/yolov4-tiny-416"
    flag_size = 416
    flag_framework = "tf"
    flag_tiny = True
    flag_model = "yolov4"
    flag_iou = 0.45
    flag_score = 0.50
    flag_dont_show = False
    flag_info = False
    flag_count = False

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    print("10%")
    # initialize deep sort
    model_filename = "model_data/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)
    print("40%")

    input_size = flag_size
    print("50%")
    saved_model_loaded = tf.saved_model.load(flag_weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures["serving_default"]
    print("80%")

    frame_num = 0
    # while video is running
    count2 = 0

    face_detector = cv2.CascadeClassifier(
        "./opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
    )  # 경로 주의
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_names = ["None", "jk", "jh", "hs"]
    faces = []

    person_in = set()
    person_noti = False
    person_time = 0
    stranger_count = 0
    stranger_in = False
    s3 = boto3.client("s3")
    base_url = "https://lunarai.s3.ap-northeast-2.amazonaws.com/"
    img_idx = 200
    while True:
        frame = recv_socket.recv_string()
        img = base64.b64decode(frame)
        np_img = np.frombuffer(img, dtype=np.uint8)
        source = cv2.imdecode(np_img, 1)
        frame = source
        count2 += 1

        if count2 % 2 == 0:

            # print("=skip=")
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y : y + h, x : x + w])
                # Check if confidence is less them 100 ==> "0" is perfect match
                if (100 - confidence) > -20 and (100 - confidence) <= 100:
                    id = face_names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = 0
                    confidence = "  {0}%".format(round(100))

                cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    str(confidence),
                    (x + 5, y + h - 5),
                    font,
                    1,
                    (255, 255, 0),
                    1,
                )
            for track in tracker.tracks:
                class_name = track.get_class()
                if class_name == "person":
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    class_name = track.get_class()

                    # draw bbox on screen
                    color = colors[0]  # int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(
                        frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color,
                        2,
                    )
                    cv2.rectangle(
                        frame,
                        (int(bbox[0]), int(bbox[1] - 30)),
                        (
                            int(bbox[0])
                            + (len(class_name) + len(str(track.track_id))) * 17,
                            int(bbox[1]),
                        ),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        class_name + "-" + str(track.track_name),
                        (int(bbox[0]), int(bbox[1] - 10)),
                        0,
                        0.75,
                        (255, 255, 255),
                        2,
                    )
            continue

        frame_num += 1
        # print("Frame #: ", frame_num)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # img = cv2.flip(frame, 1)  # Flip vertically
        # img = cv2.flip(img, 1)  # Flip vertically

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(64), int(48)),
        )

        # run detections on tflite if flag is set

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        (
            boxes,
            scores,
            classes,
            valid_detections,
        ) = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=flag_iou,
            score_threshold=flag_score,
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0 : int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0 : int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0 : int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if flag_count:
            cv2.putText(
                frame,
                # "Objects being tracked: {}".format(count),
                (5, 35),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 255, 0),
                2,
            )
            # print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [
            Detection(bbox, score, class_name, feature)
            for bbox, score, class_name, feature in zip(bboxes, scores, names, features)
        ]

        # initialize color map
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores
        )
        detections = [detections[i] for i in indices]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            class_name = track.get_class()
            if class_name == "person":
                bbox = track.to_tlbr()
                for (x, y, w, h) in faces:
                    # print(x, y, w, h, "얘가 얼굴")
                    # print(bbox[0], bbox[1], bbox[2], bbox[3], "얘가 객체스~")
                    if (
                        int(bbox[0] <= x)
                        and (int(bbox[1]) <= y)
                        and (int(bbox[2]) >= x + w)
                        and (int(bbox[3]) >= y + h)
                    ):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        id, confidence = recognizer.predict(gray[y : y + h, x : x + w])
                        # Check if confidence is less them 100 ==> "0" is perfect match
                        if (100 - confidence) > -20 and (100 - confidence) <= 100:
                            id = face_names[id]
                            confidence = "  {0}%".format(round(100 - confidence))
                            track.track_name = id
                            person_in.add(track.track_name)
                        else:
                            id = 0
                            confidence = "  {0}%".format(round(100))
                            track.track_name = id
                            person_in.add(track.track_name)

                        cv2.putText(
                            frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2
                        )
                        cv2.putText(
                            frame,
                            str(confidence),
                            (x + 5, y + h - 5),
                            font,
                            1,
                            (255, 255, 0),
                            1,
                        )

                # draw bbox on screen
                color = colors[0]  # int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2,
                )
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1] - 30)),
                    (
                        int(bbox[0])
                        + (len(class_name) + len(str(track.track_id))) * 17,
                        int(bbox[1]),
                    ),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    class_name + "-" + str(track.track_name),
                    (int(bbox[0]), int(bbox[1] - 10)),
                    0,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                # if enable info flag then print details about each track
                # if flag_info:
                #     print(
                #         "Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                #             str(track.track_id),
                #             class_name,
                #             (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                #         )
                #     )

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        # print("====================")
        # print("object : ")

        person_check = 0
        person_is_in = 0
        temp_set = person_in.copy()
        for person in temp_set:
            for track in tracker.tracks:
                if track.track_name == person:
                    person_check = 1
                    stranger_count = 0
            if person_check == 0 and person_noti == 1:
                print(f"============={person} is out")
                save_log("img_url", "out", person)
                stranger_in = 0
                person_noti = 0
                person_time = 0
                tracker.tracks = []

        person_in = set()
        have_person = 0
        print()
        for track in tracker.tracks:
            if track.get_class() == "person":
                if not track.track_name == 0:
                    person_in.add(track.track_name)
                    person_time += 1
                if person_check and person_noti == 0 and person_time >= 20:
                    print(f"================{track.track_name} is in")
                    save_log("img_url", "in", track.track_name)

                    person_noti = 1
                    person_time = 0
                have_person = 1
                print(f"[id:{track.track_id}, name:{track.track_name}]")
                if person_check == 0:
                    stranger_count += 1
                    if stranger_in == False and stranger_count == 60:
                        print("===========stranger in!  warning!")
                        cv2.imwrite(f"pic/{img_idx}.png", frame)
                        save_log("img_url", "in", "stranger")

                        s3.upload_file(
                            f"pic/{img_idx}.png", "lunarai", f"{img_idx}.png"
                        )
                        img_name = f"{img_idx}.png"

                        sendMessage(base_url + img_name)
                        img_idx += 1
                        stranger_in = True

        if have_person == 0 and stranger_in == True:
            print("===========stranger out")
            save_log("img_url", "out", "stranger")

            stranger_count = 0
            stranger_in = False

        # todo 모델 코드 작성 (현재는 흑백으로 변환되게 해둠)

        encoded, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer)
        send_socket.send(jpg_as_text)


context = zmq.Context()

send_socket = context.socket(zmq.PUB)
send_socket.bind("tcp://*:5000")

recv_socket = context.socket(zmq.SUB)
# recv_socket.connect("tcp://192.168.0.14:4000")  # 라즈베리파이 ip

recv_socket.connect("tcp://localhost:4000")  # 라즈베리파이 ip
recv_socket.setsockopt_string(zmq.SUBSCRIBE, np.compat.unicode(""))

recv_socket2 = context.socket(zmq.SUB)
recv_socket2.connect("tcp://localhost:6002")
recv_socket2.setsockopt_string(zmq.SUBSCRIBE, np.compat.unicode(""))


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

face_model_thread = threading.Thread(target=face_model, args=(recv_socket, send_socket))
process_thread = threading.Thread(target=process, args=(recv_socket2,))

face_model_thread.start()
process_thread.start()
