# USAGE
#video input
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco
##python yolo_video.py --yolo yolo-coco

import numpy as np
import argparse
import imutils
import time
import cv2
import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException



account_sid="Axxxxxxxxxxxxxxxxxxxxxxxxxxxxxx9"
auth_token="3xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx0"
client = Client(account_sid, auth_token)


ap = argparse.ArgumentParser()

ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)


try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))


except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
#Cont_e=0
Cont_z=0
Cont_b=0
Cont_g=0
Cont_h=0
Cont_c=0
Cont_d=0
Cont_w=0
Cont_s=0

while True:
        
        (grabbed, frame) = vs.read()

        
        if not grabbed:
                break
        if W is None or H is None:
                (H, W) = frame.shape[:2]

        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        
        boxes = []
        confidences = []
        classIDs = []

        
        for output in layerOutputs:
                
                for detection in output:
                        
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        
                        if confidence > args["confidence"]:
                                
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                args["threshold"])

        
        if len(idxs) > 0:
                
                for i in idxs.flatten():
                        
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                       
                        
                        text1="{}".format(LABELS[classIDs[i]])
                        cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        try:
                                sms_text = "Animal detected is {name}".format(name = LABELS[classIDs[i]])
                                message=client.api.account.messages.create(
                                body=sms_text,
                                from_='+16467988374',
                                to='+918xxxxxxxx6'
                                )
                        except TwilioRestException as ex:
                                print(ex)
                        

        cv2.imshow('name',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break



print("[INFO] cleaning up...")

vs.release()
