import boto3
import requests
from os import path
import json


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
            "description": "테스트용 데이터입니다.",
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


s3 = boto3.client("s3")

s3.upload_file("pic/test.png", "lunarai", "test.png")
base_url = "https://lunarai.s3.ap-northeast-2.amazonaws.com/"
img_name = "test.png"

sendMessage(base_url + img_name)
