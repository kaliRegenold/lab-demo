import cv2 as cv
import sys
import time

def background_removal():
    # Camera stream
    video = cv.VideoCapture(0)

    # Camera resolution (these number work for my current setup,
    # use something else as needed.)
    video.set(cv.CAP_PROP_FRAME_WIDTH, 1020)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 768)

    # Create background subtractor
    mog2 = cv.createBackgroundSubtractorMOG2(history=64, varThreshold=32, detectShadows=False)

    # Window to display
    cv.namedWindow('frame',cv.WINDOW_NORMAL)
    cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    timed = False
    if len(sys.argv) == 2:
        timed = True
        end_time = time.time() + int(sys.argv[1])

    while(True):
        # Read a single frame
        ret, frame = video.read()

        # Apply mask using background subtractor
        foreground_mask = mog2.apply(frame)

        # Display window
        cv.imshow('frame', foreground_mask)

        # Check for 'escape' key (q)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        elif timed and time.time() > end_time:
            break

    # Release video stream and distroy display window
    video.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    background_removal()
