# Basic Usage:
# python countingserver.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
        # --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel

from imutils import build_montages
from datetime import datetime
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import dlib
import time
import threading
from eventhandler import eventhandler
import subprocess

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-mW", "--montageW", type=int, default=1,
        help="montage frame width")
ap.add_argument("-mH", "--montageH", type=int, default=1,
        help="montage frame height")
ap.add_argument("-s", "--skip-frames", type=int, default=4,
        help="# of skip frames between detections")
ap.add_argument("-f", "--framerate", type=int, default=80,
        help="FPS of written footage")
ap.add_argument("-o", "--output", type=str, default="./videologs/" + 
        "tmpvid.avi",
        help="Path that video footage will be saved to")
args = vars(ap.parse_args())

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

# initialize MobilenetSSD classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the consider set for vehicles,
# the object count dictionary, and the frame dictionary
CONSIDER = set(["motorbike", "car"])
objCount = {obj: 0 for obj in CONSIDER}
frameDict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
lastActive = {}
lastActiveCheck = datetime.now()

# stores the estimated number of Pis, active checking period, and
# calculates the duration seconds to wait before making a check to
# see if a device was active
ESTIMATED_NUM_PIS = 1
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

# assign montage width and height so we can view all incoming frames
# in a single "dashboard"
mW = args["montageW"]
mH = args["montageH"]
print("[INFO] detecting: {}...".format(", ".join(obj for obj in
        CONSIDER)))

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of processed frames
totalFrames = 0

# initialize video codec variable (storage)
codec = "MJPG"

# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc(*codec)
writer = None
zeros = None

# initialize the event handler for the parking lot
# will send notifcations every time with current setting
cherry = eventhandler("eventHandlerCherry")

# measuring FPS
fps = FPS().start()

# start looping over all the frames
while True:
    # receive RPi name and frame from the RPi and acknowledge
    # the receipt
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')

    # if a device is not in the last active dictionary then it means
    # that its a newly connected device
    if rpiName not in lastActive.keys():
            print("[INFO] receiving data from {}...".format(rpiName))

    # record the last active time for the device from which we just
    # received a frame
    lastActive[rpiName] = datetime.now()

    # resize the frame to have a maximum width of 400 pixels, then
    # grab the frame dimensions and construct a blob
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    # convert from BGR to RGB for use with dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # check if the writer is None
    if writer is None:
        # store the image dimensions, initialize the video writer,
        # and construct the zeros array
        writer = cv2.VideoWriter(args["output"], fourcc, args["framerate"],
        (w, h), True)
        zeros = np.zeros((h, w), dtype="uint8")

    # initialize current status along with list of bounding box
    # rectangles returned by either the detector or trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        # set status and initialize new object trackers
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()


        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > args["confidence"]:
                        # extract the index of the class label from the
                        # detections
                        idx = int(detections[0, 0, i, 1])

                        # check to see if the predicted class is in the set of
                        # classes that need to be considered
                        if CLASSES[idx] in CONSIDER:
                                # compute the (x, y)-coordinates of the bounding box
                                # for the object
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")

                                # construct a dlib rectangle object from the bounding
                                # box coordinates and then start the dlib correlation
                                # tracker
                                tracker = dlib.correlation_tracker()
                                rect = dlib.rectangle(startX, startY, endX, endY)
                                tracker.start_track(rgb, rect)

                                # add the tracker to our list of trackers so we can
                                # utilize it during skip frames
                                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a vertical line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving left or right
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 0, 255), 2)

    # update the new frame in the frame dictionary
    frameDict[rpiName] = frame

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
           to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
           # the difference between the x-coordinate of the *current*
           # centroid and the mean of *previous* centroids will tell
           # us in which direction the object is moving (negative for
           # left and positive for right)
           x = [c[0] for c in to.centroids]
           direction = centroid[0] - np.mean(x)
           to.centroids.append(centroid)

           # check to see if the object has been counted or not
           # the +/-20 was added as a tolerance to avoid false 
           # counts when cars are immediately recognized
           if not to.counted:
               # if the direction is negative (indicating the object
               # is moving left) AND the centroid is to the left
               # of the line, count the car
               if direction < -20 and centroid[0] < w // 2:
                   cherry.entry()
                   to.counted = True

               # if the direction is positive (indicating the object
               # is moving right) AND the centroid is 
               # to the right of the line, count the car as leaving
               elif direction > 20 and centroid[0] > w // 2:
                   cherry.exit()
                   to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Count", cherry.count), 
        ("Status", status),
    ]

    # draw the sending device name on the frame
    cv2.putText(frame, rpiName, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # draw timestamp on frame
    cv2.putText(frame, time.strftime("%H:%M"), (125, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
       text = "{}: {}".format(k, v)
       cv2.putText(frame, text, (10, h - ((i * 20) + 20)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # build a montage using images in the frame dictionary
    montages = build_montages(frameDict.values(), (w, h), (mW, mH))

    # display the montage(s) on the screen
    for (i, montage) in enumerate(montages):
            cv2.imshow("WKU Occupancy Monitoring ({})".format(i),
                    montage)

    # construct the final output frame
    output = np.zeros((h, w, 3), dtype="uint8")
    output[0:h, 0:w] = frame

    # write the output frame to file
    writer.write(output)

    # detect any kepresses
    key = cv2.waitKey(1) & 0xFF

    # if current time *minus* last time when the active device check
    # was made is greater than the threshold set then do a check
    if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
            # loop over all previously active devices
            for (rpiName, ts) in list(lastActive.items()):
                    # remove the RPi from the last active and frame
                    # dictionaries if the device hasn't been active recently
                    if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                            print("[INFO] lost connection to {}".format(rpiName))
                            lastActive.pop(rpiName)
                            frameDict.pop(rpiName)

            # set the last active check time as current time
            lastActiveCheck = datetime.now()

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
            break

    # if the 'm' key was pressed, create new thread to allow
    # for manual count setting
    if key == ord("m"):
        manualT = threading.Thread(target=cherry.askManualSet)
        manualT.start()

    #increment total number of frames processed so far
    totalFrames += 1

    # For FPS testing
    fps.update()

# cleanup for display, writer, and logs
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
writer.release()
# run video log handler to organize footage
subprocess.run(["bash", "vloghandler"])

# Stop fps counting tool and print the avg FPS
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
