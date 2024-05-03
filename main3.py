
# STEP1
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from insightface.app import FaceAnalysis


# STEP2
compare = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
compare.prepare(ctx_id=0, det_size=(640, 640))

app = FastAPI()

# STEP3
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile, file2: UploadFile):
    contents = await file.read()  
    contents2 = await file2.read()  
    
    nparr = np.fromstring(contents, np.uint8) 
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 

    nparr2 = np.fromstring(contents2, np.uint8)  
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)  

    faces = compare.get(img)
    faces2 = compare.get(img2)

    feat1 = np.array(faces[0].normed_embedding, dtype=np.float32) 
    feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32) 
#STEP4
    sims = np.dot(feat1, feat2.T)
#STEP5
    return float(sims)




