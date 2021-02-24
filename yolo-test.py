import cv2
import numpy as np
#
# img = cv2.imread('cars.png')
# cv2.imshow('Image', img)
# cv2.waitKey(-1)

size = 320

def findObjects(outputs, img):
    height, width, center = img.shape
    b_box = []
    classIds = []
    confs = []
    # 2, 3, 5, 7, 1
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            if classId not in [2,3,5,7,1]:
                continue
            confidence = scores[classId]

            if confidence > THRESHOLD:
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int((detection[0] * width) - w/2), int((detection[1] * height) - h/2)

                b_box.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    # print(len(b_box))
    # print(confs)
    indices = cv2.dnn.NMSBoxes(b_box, confs, THRESHOLD, NMS_THRESHOLD)
    # print(indices)
    for box in indices:
        bound_box = b_box[box[0]]
        x,y,w,h = bound_box[0], bound_box[1], bound_box[2], bound_box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 2)

    # cv2.imshow('Image', img)
    # cv2.waitKey(-1)


with open('coco.names', 'r') as f:
    classnames = f.read().splitlines()

config = 'cfg'
weights = 'yolov3.weights'
THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
vid = cv2.VideoCapture('highway1.mp4')

frame_count = 0
while(vid.isOpened()):
    ret, frame = vid.read()
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_count % 20 == 0:
        blob = cv2.dnn.blobFromImage(frame, 1/255, (size, size), [0,0,0], 1, crop=False)
        net.setInput(blob)

        layernames = net.getLayerNames()
        # print(layernames)
        # print(net.getUnconnectedOutLayers())
        outputnames = [layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]
        # print(outputnames)

        outputs = net.forward(outputnames)
        findObjects(outputs, frame)
        cv2.imshow('frame', frame)
        frame_count = 0

