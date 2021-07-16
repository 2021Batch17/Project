
# Import the necessary packages
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
from intel.yolo_v3_params import YoloV3Params
from intel.yolo_v3 import Yolo_V3
from imutils.video import VideoStream
from pyimagesearch.utils.conf import Conf
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import RPi.GPIO as GPIO
from twilio.rest import Client

account_sid = 'Axxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx4'
auth_token = '0xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxb'
client = Client(account_sid, auth_token)

count=0
buzzer = 18
GPIO.setmode(GPIO.BOARD)
GPIO.setup(buzzer,GPIO.OUT)
GPIO.output(buzzer,GPIO.HIGH)
text = 'None'
text1 = 'None1'

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", default="/home/pi/Desktop/Project/Code/config/config.json", help="Path to the input configuration file")
ap.add_argument("-i", "--input", help="path to the input video file")
args = vars(ap.parse_args())


conf = Conf(args["conf"])


LABELS = open(conf["labels_path"]).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))


plugin = IEPlugin(device="MYRIAD")


print("[INFO] Loading the models...")
net = IENetwork(model=conf["xml_path"], weights=conf["bin_path"])


print("[INFO] Preparing inputs...")
inputBlob = next(iter(net.inputs))


net.batch_size = 1
(n, c, h, w) = net.inputs[inputBlob].shape


if args["input"] is None:
    print("[INFO] Starting the video stream...")
    #vs = VideoStream(src=0).start()
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)


else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(os.path.abspath(args["input"]))


print("[INFO] Loading the model to the plugin...")
execNet = plugin.load(network=net, num_requests=1)
#fps = FPS().start()


while True:
    
    orig = vs.read()
    orig = orig[1] if args["input"] is not None else orig
    
    if args["input"] is not None and orig is None:
        break
    
    orig = imutils.resize(orig, width=500)
    frame = cv2.resize(orig, (w, h))
    
    frame = frame.transpose((2, 0, 1))
    frame = frame.reshape((n, c, h, w))
    
    output = execNet.infer({inputBlob: frame})
    objects = []
    
    for (layerName, outBlob) in output.items():
        
        layerParams = YoloV3Params(net.layers[layerName].params, outBlob.shape[2])
        
        objects += Yolo_V3.parse_yolo_region(outBlob,frame.shape[2:], orig.shape[:-1], layerParams,
                                                conf["prob_threshold"])
    
    for i in range(len(objects)):
       
        #print("[INFO] No of object: {:.2f}".format(objects[i]["confidence"]))
        #print("[INFO] No of object: {:.2f}",i)
        if objects[i]["confidence"] == 0:
            continue
        # Loop over remaining objects
        for j in range(i + 1, len(objects)):
            
            #print("[INFO] IoU: {:.2f}".format(Yolo_V3.intersection_over_union(objects[i], objects[j])))
            #print("[INFO] Threshold: {:.2f}".format(conf["iou_threshold"]))
            if Yolo_V3.intersection_over_union(objects[i], objects[j]) > conf["iou_threshold"]:
                objects[j]["confidence"] = 0
    
                
    objects = [obj for obj in objects if obj['confidence'] >= conf["prob_threshold"]]
    
    (endY, endX) = orig.shape[:-1]
   
    for obj in objects:
        
        if obj["xmax"] > endX or obj["ymax"] > endY or obj["xmin"] < 0 or obj["ymin"] < 0:
            continue
        
        label = "{}: {:.2f}%".format(LABELS[obj["class_id"]], obj["confidence"] * 100)
        text = "{}".format(LABELS[obj["class_id"]])
        
        y = obj["ymin"] - 15 if obj["ymin"] - 15 > 15 else \
            obj["ymin"] + 15
        
        cv2.rectangle(orig, (obj["xmin"], obj["ymin"]), (obj["xmax"], obj["ymax"]), COLORS[obj["class_id"]], 2)
        cv2.putText(orig, label, (obj["xmin"], y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[obj["class_id"]], 3)
      
    cv2.imshow("YOLOv3", orig)
    key = cv2.waitKey(1) & 0xFF
    
    if(text =='Person' or text == 'Cow' or text == 'Bear' or text == 'Bull' or text == 'Cat' or text == 'Cheetah' or text == 'Dog' or text == 'Elephant' or text == 'Giraffe' or text == 'Horse' or text == 'Monkey' or text == 'Pig' or text == 'Sheep' or text == 'Zebra' or text == 'Lion'):
        GPIO.output(buzzer,GPIO.LOW)
        #print("Object detected")
        count+=1
    else:
        GPIO.output(buzzer,GPIO.HIGH)
        count=0
    if(count==5):
        #print("new object detected")
        sms_text = 'Object detected is',text
        print(sms_text)
        #message = client.messages.create(body=sms_text,from_='15037665773',to='+919686520517')
    if(text1!=text):
        count=0
        text1=text
    
    if key == ord("q"):
        GPIO.output(buzzer,GPIO.HIGH)
        break
   

#print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
# Stop the video stream and close any open windows1
vs.stop() if args["input"] is None else vs.release()
cv2.destroyAllWindows()
