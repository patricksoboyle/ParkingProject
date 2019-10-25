# Basic Usage:
# python client.py -s SERVER_IP

from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
        help="ip address of the server to which the client will connect")
ap.add_argument("-rw", "--res-width", type=int, default = 272,
        help="resolution width of captured footage")
ap.add_argument("-rh", "--res-height", type=int, default=208,
        help="resolution height of captured footage")
ap.add_argument("-f", "--framerate", type=int, default=80,
        help="framerate of captured footage")
args = vars(ap.parse_args())

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
        args["server_ip"]))

# get the host name and initialize the video stream
rpiName = socket.gethostname()
vs = VideoStream(usePiCamera=True, resolution=(args["res_width"],
   args["res_height"]), framerate=args["framerate"]).start()

# camera warmup
time.sleep(2.0)
 
while True:
        # read the frame from the camera and send it to the server
        frame = vs.read()
        sender.send_image(rpiName, frame)
