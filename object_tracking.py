#############################################################################
# Object Tracking for 3D printing to detect failures
#
# Description: Tracking the print head to get the position in real case
#   and compare it to the theoretical position that is taken from the gcode
#
# References:
#   - https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
#   - https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
#############################################################################

import argparse
import time
from collections import deque

import cv2
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return int(rightMin + (valueScaled * rightSpan))


def track_video(video_stream, tracker, tracker_name, video):
    # initialize the bounding box coordinates of the tracked object
    init_bb = None

    # initialize the FPS throughput estimator
    fps = None

    # initialize the list of tracked points, the frame counter and the coordinate deltas
    buffer = 800
    pts = deque(maxlen=buffer)

    # loop over frames from the video stream
    while True:
        # grab the current frame
        frame = video_stream.read()
        frame = frame[1] if video else frame

        # check if the video stream has ended
        if frame is None:
            break
        # resize the frame and grab the frame dimensions
        # frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if init_bb is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            x = y = w = h = 1
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # update the FPS counter
            fps.update()
            fps.stop()

            x_real = translate(x, 270, 911, 0, 190)
            y_real = 190 - translate(y, 28, 645, -2, 190)

            # display tracking line
            center = (x + int(w / 2), y + int(h / 2))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)
            # loop over the set of tracked points
            for i in np.arange(1, len(pts)):
                # if either of the tracked points are None, ignore them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and draw the connecting lines
                thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness)

            # initialize the set of information we'll be displaying on the frame
            try:
                fps_value = fps.fps()
            except ZeroDivisionError:
                fps_value = 0
            info = [
                ("Tracker", tracker_name),
                ("Position", "(" + str(x_real) + ", " + str(y_real) + ")"),
                ("FPS", "{:.2f}".format(fps_value) if success else 0),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 0, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            init_bb = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
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
    # parse command line arguments
    args = parse_arguments()

    # grab the appropriate object tracker using our dictionary of OpenCV object tracker objects
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
    # close window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
