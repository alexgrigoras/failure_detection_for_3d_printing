###########################################################################
# Object Tracking for 3D printing to detect failures
# Software adapter from
# https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
# https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
###########################################################################

from collections import deque

import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


def track_video(video_stream, tracker, tracker_name, video):
    # initialize the bounding box coordinates of the object we are going
    # to track
    init_bb = None

    # initialize the FPS throughput estimator
    fps = None

    # initialize the list of tracked points, the frame counter,
    # and the coordinate deltas
    buffer = 300
    pts = deque(maxlen=buffer)
    counter = 0
    (dX, dY) = (0, 0)
    direction = ""

    # loop over frames from the video stream
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = video_stream.read()
        frame = frame[1] if video else frame
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        #frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if init_bb is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            x = 1
            y = 1
            w = 1
            h = 1
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
            # update the FPS counter
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", tracker_name),
                #("Success", "Yes" if success else "No"),
                ("Position", "(" + str(x) + ", " + str(y) + ")"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            #######################################################################

            center = (x+int(w/2), y+int(h/2))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)

            # loop over the set of tracked points
            for i in np.arange(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # check to see if enough points have been accumulated in
                # the buffer
                if counter >= 10 and i == 1 and pts[-10] is not None:
                    # compute the difference between the x and y
                    # coordinates and re-initialize the direction
                    # text variables
                    dX = pts[-10][0] - pts[i][0]
                    dY = pts[-10][1] - pts[i][1]
                    (dirX, dirY) = ("", "")
                    # ensure there is significant movement in the
                    # x-direction
                    if np.abs(dX) > 20:
                        dirX = "East" if np.sign(dX) == 1 else "West"
                    # ensure there is significant movement in the
                    # y-direction
                    if np.abs(dY) > 20:
                        dirY = "North" if np.sign(dY) == 1 else "South"
                    # handle when both directions are non-empty
                    if dirX != "" and dirY != "":
                        direction = "{}-{}".format(dirY, dirX)
                    # otherwise, only one direction is non-empty
                    else:
                        direction = dirX if dirX != "" else dirY

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness)

            #######################################################################

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            init_bb = cv2.selectROI("Frame", frame, fromCenter=False,
                                   showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, init_bb)
            fps = FPS().start()

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break



def parse_arguments():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    return args


def main():
    args = parse_arguments()

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

    vs = None
    # if a video path was not supplied
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])

    track_video(vs, tracker, args["tracker"], args["video"])

    vs.release()
    # close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
