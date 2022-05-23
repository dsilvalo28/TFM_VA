from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import pandas as pd


lista_boxes = []
lista_etiquetas = []
lista_frames = []


def calcular_movimiento(tag, box1, box2, n_frames1, n_frames2):
    print(tag)
    n_frames = n_frames2 - n_frames1

    (x1, y1, w1, h1) = box1
    (x2, y2, w2, h2) = box2
    movi_x = (x2 - x1) / n_frames
    movi_y = (y2 - y1) / n_frames
    movi_w = (w2 - w1) / n_frames
    movi_h = (h2 - h1) / n_frames

    coordenadas = []
    for i in range(n_frames):
        lista_etiquetas.append(tag)
        lista_boxes.append((x1 + movi_x * i, y1 + movi_y * i,
                            w1 + movi_w * i, h1 + movi_h * i))
        lista_frames.append(n_frames1 + i)

        coordenadas.append((tag,
                           (x1 + movi_x * i, y1 + movi_y * i,
                            w1 + movi_w * i, h1 + movi_h * i),
                           n_frames1 + i))

    i = n_frames2
    lista_etiquetas.append(tag)
    lista_boxes.append((x1 + movi_x * i, y1 + movi_y * i,
                        w1 + movi_w * i, h1 + movi_h * i))
    lista_frames.append(n_frames1 + i)

    coordenadas.append((tag, (x1 + movi_x * i, y1 + movi_y * i,
                              w1 + movi_w * i, h1 + movi_h * i), i))
    return coordenadas


def calcular_rectangulos(posibles_frames, ):
    if n_frames in posibles_frames:
        df_aux = medidas[medidas["Frame"] == n_frames]
        rectangulos = list(df_aux["Box"])
        for rectangulo in rectangulos:
            (x, y, w, h) = [int(float(v)) for v in rectangulo.split(", ")[1:-1]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

videoPath = "dashcam_boston.mp4"
vs = cv2.VideoCapture(videoPath)

medidas = pd.read_csv("boxes.csv")
boxes = []

etiquetas_finales = []
dic_etiquetas = {}
n_frames = 1
posibles_frames = list(medidas["Frame"])
print(posibles_frames)
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    success, frame = vs.read()
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame, width=600)
    if n_frames in posibles_frames:
        df_aux = medidas[medidas["Frame"] == n_frames]
        rectangulos = list(df_aux["Box"])
        for rectangulo in rectangulos:
            (x, y, w, h) = [int(float(v)) for v in rectangulo[1:-1].split(", ")]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # for box in boxes:
    #     (x, y, w, h) = [int(v) for v in box]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        boxes = cv2.selectROIs("Frame", frame, fromCenter=False, showCrosshair=True)

        for box in boxes:
            print(box)
            etiqueta = input("Introduce etiqueta")
            if etiqueta in dic_etiquetas.keys():
                etiquetas_finales.append(
                    calcular_movimiento(
                        etiqueta, dic_etiquetas.get(etiqueta)[0], box,
                        dic_etiquetas.get(etiqueta)[1], n_frames))

                print(etiquetas_finales)
            else:
                dic_etiquetas[etiqueta] = (box, n_frames)

        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
    n_frames += 1
    if key == ord("q"):
        break

vs.release()
# close all windows
cv2.destroyAllWindows()

d = {'Etiqueta': lista_etiquetas, 'Frame': lista_frames, 'Box': lista_boxes}
df = pd.DataFrame(data=d)

print(df.head())

# df.to_csv("boxes.csv", index=False),
