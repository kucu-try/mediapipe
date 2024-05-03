# # STEP1
from transformers import pipeline


# # STEP2
vision_classifier = pipeline(model="google/vit-base-patch16-224")


# STEP1
from fastapi import FastAPI, File, UploadFile


app = FastAPI()

# 바이츠를 읽어온다.
@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


import numpy as np
import cv2
from PIL import Image
import io
# 파일을 보낼거야라는 뜻(한번더 바이츠로 줘야 읽을 수 있다.)(비동기적인 특성을 가질 수 있다.)
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()  #텍스트 형식으로 바뀜

###### opencv
    # nparr = np.fromstring(content, np.unit8)  # 텍스트 형식을 파일 형식으로 바꿔줘야한다.
# # http형식으로 파일을 보낼 때 텍스트 인코딩된걸 파일 형식으로 변경 후 보내줘야한다.
    # img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) #디코딩 처리해야한다.



    # cv2.imread('images\irin.jpg')
    # file open
    # image decoding 이 포함됨 

    #####PIL 
    ioBytes = io.BytesIO(contents)
    img = Image.open(ioBytes)

        # # STEP4
    preds = vision_classifier(images=img)

    # # STEP5
    preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]

    return preds

