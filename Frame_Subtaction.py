#Frame subtraction
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int, default=800, help="minimum area size")
args = vars(ap.parse_args())
vs = VideoStream(src=0).start()
time.sleep(2.0)

firstFrame = None

while True:
	
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	if frame is None:
		break
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
	if firstFrame is None:
		firstFrame = gray
		continue		
	
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=3)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	for c in cnts:
		if cv2.contourArea(c) < args["min_area"]:
			continue
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.imshow("Image Feed", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
