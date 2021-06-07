import cv2
import numpy
import time
import pyttsx3
import os
import numpy as np
import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(640, 480))

os.system("pacmd set-card-profile bluez_card.41_42_71_EB_42_A6 a2dp_sink")
os.system("pacmd set-default-sink bluez_sink.41_42_71_EB_42_A6.a2dp_sink")
# 
def onStart(name):
#    print 'starting', name
   pass
def onWord(name, location, length):
#    print 'word', name, location, length
   pass
def onEnd(name, completed):
#    print 'finishing', name, completed
   pass

engine = pyttsx3.init('espeak')
engine.setProperty('rate',95)     # setting up new voice rate
engine.setProperty('volume',1) 

engine.connect('started-utterance', onStart)
engine.connect('started-word', onWord)
engine.connect('finished-utterance', onEnd)


# 
INPUT_FILE='/home/pi/Desktop/images/654321images146.jpg'
LABELS_FILE='/home/pi/darknet/data/yolof.names'
CONFIG_FILE='/home/pi/darknet/cfg/yolov3f.cfg'
WEIGHTS_FILE='/home/pi/darknet/yolov3f_3000.weights'

CONFIDENCE_THRESHOLD=0.3

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")


net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    # Capture frame-by-frame
# image = cv2.imread(INPUT_FILE)
    image = frame.array
    (H, W) = image.shape[:2]




    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()


    print("[INFO] YOLO took {:.6f} seconds".format(end - start))


    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        
    # loop over each of the detections
        for detection in output:
    # extract the class ID and confidence (i.e., probability) of
    # the current object detection
        
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
#             print(confidence)
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE_THRESHOLD:
    # scale the bounding box coordinates back relative to the
    # size of the image, keeping in mind that YOLO actually
    # returns the center (x, y)-coordinates of the bounding
    # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
            
    # use the center (x, y)-coordinates to derive the top and
    # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

    # update our list of bounding box coordinates, confidences,
    # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
            

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:
    # loop over the indexes we are keeping
        for i in idxs.flatten():
    # i=0
    # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            d=0.6840427896962948326805849707893671456322795994*pow(w*h,0.24606769460876075199171728705294118326193913388)
            text = "{}: {:.4f} {} inches".format(LABELS[classIDs[i]], confidences[i], str(round(d,2)))
            print(text)
            
            
            engine.say(LABELS[classIDs[i]])
            engine.say(str(round(d,2))+" inches")
            print(str(round(d,2))+" inches")
            engine.runAndWait()
            engine.stop()
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
# 
#             # show the output image
    cv2.imshow("Detection", image)
    if cv2.waitKey(1000) & 0xff == ord('q'):
        break
    rawCapture.truncate(0)
    
camera.close()
cv2.destroyAllWindows()
 
