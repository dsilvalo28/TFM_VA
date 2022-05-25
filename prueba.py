from imutils.video import VideoStream
import argparse
import imutils
import pandas as pd
import cv2


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


def ordenar_diccionario_trackers(dic_t, n_tracker_eliminar, n_trackers_total):
    print(dic_t)

    for i in range(n_tracker_eliminar, n_trackers_total):
        dic_t[i] = dic_t.get(i + 1)

    dic_t.pop(i)
    return dic_t


def actualizar_trackers(dic_t, track, n_trackers_total):
    nuevo_tracker = cv2.MultiTracker_create()
    print(dic_t)

    for i in range(n_trackers_total + 1):
        ultimo_box = dic_t.get(i)[1][-1]
        ultimo_frame = dic_t.get(i)[2][-1]

        nuevo_tracker.add(track, ultimo_frame, ultimo_box)

    return nuevo_tracker


def calcular_rectangulos(posibles_frames):
    if n_frames in posibles_frames:
        df_aux = medidas[medidas["Frame"] == n_frames]
        rectangulos = list(df_aux["Box"])
        for rectangulo in rectangulos:
            (x, y, w, h) = [int(float(v)) for v in rectangulo.split(", ")[1:-1]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


# initialize a dictionary that maps strings to their correspondings
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()
tracker = OPENCV_OBJECT_TRACKERS["csrt"]()

videoPath = "dashcam_boston.mp4"
vs = cv2.VideoCapture(videoPath)

medidas = pd.read_csv("boxes.csv")
boxes = []

etiquetas_finales = []
dic_etiquetas = {}
dic_trackers = {}
dic_final = {}
list_boxes = []
list_frames = []
n_frames = 1
n_tracker = 0
posibles_frames = list(medidas["Frame"])

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    success, frame = vs.read()
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame, width=600)
    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    print(trackers.getObjects())

    (success, boxes) = trackers.update(frame)
    print(trackers.getObjects())

    # loop over the bounding boxes and draw then on the frame
    cont_trackers = 0
    for box in boxes:
        dic_trackers.get(cont_trackers)[1].append(box)
        dic_trackers.get(cont_trackers)[2].append(n_frames)

        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cont_trackers += 1
    # show the output frame
    cv2.imshow("Frame", frame)

    if n_frames == 1:
        key = cv2.waitKey(300) & 0xFF
    else:
        key = cv2.waitKey(200) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        boxes = cv2.selectROIs("Frame", frame, fromCenter=False,
                               showCrosshair=True)
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        for box in boxes:
            etiqueta = input("Introduce etiqueta")
            if etiqueta in dic_etiquetas.keys():
                n_tracker_eli = dic_etiquetas.get(etiqueta)[0]
                dic_final[etiqueta] = (dic_trackers.get(n_tracker_eli)[1], dic_trackers.get(n_tracker_eli)[2])

                dic_trackers = ordenar_diccionario_trackers(dic_trackers, n_tracker_eli, n_tracker)
                trackers = actualizar_trackers(dic_trackers, tracker, n_tracker)

            else:

                dic_trackers[n_tracker] = (etiqueta, list([box]), list([n_frames]))
                dic_etiquetas[etiqueta] = (n_tracker, box, n_frames)
                print(box)

                trackers.add(tracker, frame, box)
                n_tracker += 1
    n_frames += 1

    if key == ord("q"):
        break

vs.release()
# close all windows
cv2.destroyAllWindows()
