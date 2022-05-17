from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2


videoPath = "dashcam_boston.mp4"
vs = cv2.VideoCapture(videoPath)

boxes = []
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

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        boxes.append(cv2.selectROI("Frame", frame, fromCenter=False,
                                   showCrosshair=True))
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker

    if key == ord("q"):
        break

vs.release()
# close all windows
cv2.destroyAllWindows()
