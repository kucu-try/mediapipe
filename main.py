# STEP 1
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import shutil
import os
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import numpy as np
import cv2
from PIL import Image

# STEP 2
face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face.prepare(ctx_id=0, det_size=(640, 640))

# 사전 등록된 얼굴 데이터 로드
def load_all_face_embeddings(directory="images") -> Dict[str, np.ndarray]:
    known_faces = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        img = cv2.imread(path)
        faces = face.get(img)
        if faces:
                # Store the embedding of the first detected face
                known_faces[filename] = np.array(faces[0].normed_embedding, dtype=np.float32)
    return known_faces


app = FastAPI()

# images 폴더 경로 확인 및 생성
image_folder = "images"
os.makedirs(image_folder, exist_ok=True)


@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):

    # STEP 3
    contents1 = await file1.read()
    contents2 = await file2.read()
    nparr1 = np.fromstring(contents1, np.uint8)
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    nparr2 = np.fromstring(contents2, np.uint8)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # STEP 4
    faces1 = face.get(img1)
    faces2 = face.get(img2)
    print(len(faces1))
    print(len(faces2))

    # # STEP 5
    feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
    feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
    sims = np.dot(feat1, feat2.T)
    return {'similarity':float(sims)}


@app.post("/register/")
async def create_user(name: str = Form(...), photo: UploadFile = File(...)):
    # 파일 확장자 추출
    extension = Path(photo.filename).suffix
    # 입력받은 이름으로 파일 이름 생성
    safe_name = "".join(char for char in name if char.isalnum())  # 특수문자 제거, 알파벳과 숫자만 허용
    filename = f"{safe_name}{extension}"
    # 파일을 저장할 경로 지정
    file_location = os.path.join(image_folder, filename)
    # 파일을 디스크에 저장
    with open(file_location, "wb") as file_object:
        shutil.copyfileobj(photo.file, file_object)
    # 간단한 응답 메시지 반환
    return {"message": f"User {name} successfully registered!", "filename": filename}


@app.post("/identify/")
async def identify_face(file: UploadFile):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # 입력받은 이미지 처리
    faces = face.get(img)
    # error handle
    if not faces:
        raise HTTPException(status_code=404, detail="No faces detected in the uploaded image.")
    # 가장 유사한 얼굴 찾기
    best_match = None
    highest_similarity = -1
    registered_faces = load_all_face_embeddings()
    confirm_standard = 0.5
    for name, embedding in registered_faces.items():
        feat = np.array(faces[0].normed_embedding, dtype=np.float32)
        sims = np.dot(feat, embedding.T)
        # 유사도가 기준을 통과했을 때 등록
        if float(sims) >= confirm_standard:
            if sims > highest_similarity:
                highest_similarity = float(sims)
                best_match = name
    if highest_similarity == -1:
        return {"message":"보유하신 데이터와 같은 face데이터가 없습니다."}
    else:
        return {"message": "유사한 데이터를 발견!", "가장 유사한 데이터": best_match, "유사도": highest_similarity}