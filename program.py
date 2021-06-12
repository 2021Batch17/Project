from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from playsound import playsound
from twilio.rest import Client

account_sid = 'AXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX9'
auth_token = '9XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXd'
client = Client(account_sid, auth_token)
count=0
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to input video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-a", "--min-area", type=int, default=800, help="minimum area size")
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

if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["video"])
writer = None
(W, H) = (None, None)
firstFrame = None
while True:
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        if frame is None:
                break
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)
        if firstFrame is None:
                firstFrame = gray
                continue
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=3)
        cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
                if cv2.contourArea(c) <args["min_area"]:
                        continue
                try:
                        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                               else cv2.CAP_PROP_FRAME_COUNT
                        total = int(vs.get(prop))
                        print("[INFO] {} total frames in video".format(total))
                except:
                        print("[INFO] could not determine # of frames in video")
                        print("[INFO] no approx. completion time can be provided")
                        total = -1
                enter=0
                while True:
                        if args.get("video", None) is None:
                                frame = vs.read()
                        else:
                                (grabbed,frame) = vs.read()
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
                                if enter==1:
                                        enter=0
                                        break                
                                for i in idxs.flatten():
                                       
                                       (x, y) = (boxes[i][0], boxes[i][1])
                                        (w, h) = (boxes[i][2], boxes[i][3])

                                        
                                        color = [int(c) for c in COLORS[classIDs[i]]]
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                        text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                        
                                        text1="{}".format(LABELS[classIDs[i]])
                                       
                                        text == ''
                                        if (text1 == 'person'):
                                                print("person")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1

                                        if (text1 == 'pig'):
                                                print("Pig")                               
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1                                

                                        if (text1 == 'bird'):
                                                print("Bird")                               
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1


                                        if (text1 == 'horse'):
                                                print("horse")                               
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1

                                        if (text1 == 'elephant'):
                                                print("Elephant")                               
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1

                                   
                                        if (text1 == 'bear'):
                                                print("bear")                            
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                playsound('sound.mp3')
                                                count+=1

                                                
                                        if (text1 == 'zebra'):
                                                print("zebra")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                playsound('sound.mp3')
                                                count+=1

                                                                                   
                                        if (text1 == 'giraffe'):
                                                print("giraffe")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1

                                        if (text1 == 'cat'):
                                                print("cat")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1


                                        if (text1 == 'dog'):
                                                print("dog")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                playsound('sound.mp3')
                                                count+=1


                                        if (text1 == 'cow'):
                                                print("cow")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1


                                        if (text1 == 'sheep'):
                                                print("sheep")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1

                                        if (text1 == 'bull'):
                                                print("Bull")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1

                                        if (text1 == 'cheetah'):
                                                print("Cheetah")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1

                                        if (text1 == 'monkey'):
                                                print("Monkey")
                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                count+=1
                                                 if len(idxs) == 0:
                                count=0
                                print("count is zero")
                                text1=''
                        cv2.imshow('name',frame)
                        if(count>1):
                                playsound('sound.mp3')
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        if(count==2):
                                print("new object detected")
                               sms_text = 'Animal detected is',LABELS[classIDs[i]]
                                message = client.messages.create(body=sms_text,from_='+16XXXXXXXXX',to='+91XXXXXXXXXXX')
        cv2.imshow('name',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
print("[INFO] cleaning up...")
vs.release()



        
