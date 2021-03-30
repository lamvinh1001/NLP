import numpy as np
from cv2 import cv2



def feature_cap(model, video: str):
    # số frame ảnh muốn lấy của video
    num = 40
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = length//num
    t = 0
    fr = []
    fe = None
  # lay frames anh tu video
    while cap.isOpened():
        ret, frame = cap.read()
        if (t == num):
            break
        if not ret or (t == step*num):
            break
        if(t % step == 0):
            frame = cv2.resize(frame, (224, 224))
            frame = frame.reshape(1, 224, 224, 3)
            fr.append(frame)
        t += 1
  # lay feature tu anh va luu vao file
    for j in range(len(fr)):
        predict = model.predict(fr[j])

        if(j == 0):
            fe = predict
        else:
            fe = np.concatenate((fe, predict), axis=0)
    return fe
