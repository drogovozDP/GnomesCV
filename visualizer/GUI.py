import time

import cv2


class GUI:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.running = True

    def view(self, frame):
        cv2.imshow('Frame', frame)

    def controller(self):
        if cv2.waitKey(25) & 0xFF == ord('q'):
            self.running = False

    def run(self):
        assert self.cap.isOpened(), "Video is not opened"
        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            self.view(frame) if ret else None
            self.controller()
        self.cap.release()
        cv2.destroyAllWindows()
