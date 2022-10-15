import time
from functools import lru_cache

import cv2
import numpy as np

from visualizer.constants import *


class Graph:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_text = GRAPH["text"]
        self.thickness = 2
        self.font_scale = 1


    @lru_cache
    def resize(self, ratio):
        return int(self.w * ratio[0]), int(self.h * ratio[1])

    def put_text(self, canvas, text, left_w, bottom_text):
        cv2.putText(canvas, text, (left_w, bottom_text), self.font,
                    self.font_scale, self.color_text, self.thickness, cv2.LINE_AA)

    def sub_plot(self, canvas, ratio):
        w, h = self.resize(ratio)
        sub_h, sub_w, _ = canvas.shape
        sub_ratio = 0.15
        text_box_size = 4
        pad = min(sub_w * sub_ratio, sub_h * sub_ratio)
        sub_w, sub_h = int(sub_w - pad), int(sub_h - pad)
        pad_w, pad_h = (w - sub_w) // 2, (h - sub_h) // 2
        sub_h -= pad_h * (text_box_size - 1)
        sub_canvas = np.zeros((sub_h, sub_w, 3))
        return sub_canvas, sub_w, sub_h, pad_w, pad_h, text_box_size

    def plot(self, data, ratio, *args):
        w, h = self.resize(ratio)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        return canvas


class GraphClasses(Graph):
    def __init__(self, *args):
        super(GraphClasses, self).__init__(*args)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_text = GRAPH_CLASSES["text"]
        self.color_square = GRAPH_CLASSES["square"]
        self.color_rect = GRAPH_CLASSES["rect"]
        self.thickness = 2
        self.font_scale = 1

    def plot(self, data, ratio):
        canvas = super(GraphClasses, self).plot(data, ratio)
        sub_canvas, sub_w, sub_h, pad_w, pad_h, text_box_size = self.sub_plot(canvas, ratio)
        sub_canvas += GRAY
        w, h = self.resize(ratio)
        step_w = w // len(data)
        bottom_text = int(sub_h * 0.95)
        bottom_bar = int(sub_h * 0.8)
        top_bar = int(sub_h * 0.75)
        offset_w = step_w // 2
        for i in range(len(data)):
            left_w = int(step_w * i + offset_w)
            self.put_text(sub_canvas, f"{i+1}", left_w, bottom_text)
            class_rate = int(bottom_bar - top_bar * data[i])
            cv2.rectangle(sub_canvas, (left_w, bottom_bar), (left_w + offset_w // 2, class_rate), self.color_rect, -1)
            cv2.rectangle(sub_canvas, (left_w, bottom_bar), (left_w + offset_w // 2, int(bottom_bar - top_bar)),
                          self.color_square, self.thickness)

        # union subcanvas and canvas
        canvas[pad_h:h - pad_h * text_box_size, pad_w:w - pad_w] = sub_canvas

        return canvas


class GraphContinuous(Graph):
    def __init__(self, *args):
        super(GraphContinuous, self).__init__(*args)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.data = []
        self.color_absis = GRAPH_CONTINUOUS["absis"]
        self.color_text = GRAPH_CONTINUOUS["text"]
        self.color_graph = GRAPH_CONTINUOUS["graph"]
        self.thickness = 2
        self.font_scale = 1
        self.max_h = GRAPH_CONTINUOUS["max"]
        self.min_h = GRAPH_CONTINUOUS["min"]

    def plot(self, x, ratio, name):
        # prepare canvas and subcanvas
        canvas = super(GraphContinuous, self).plot(x, ratio)
        sub_canvas, sub_w, sub_h, pad_w, pad_h, text_box_size = self.sub_plot(canvas, ratio)
        w, h = self.resize(ratio)

        # update data
        self.data.append(x)
        if len(self.data) > sub_w:
            self.data.pop(0)

        # plot graph
        absis = sub_h // 2
        data = np.array(self.data)
        data_max = data.max()
        data_min = data.min()
        max_y = max(data_max, self.max_h)
        min_y = min(data_min, self.min_h)
        graph_ratio = sub_h / (max_y - min_y)
        data = max_y - data
        data = data * graph_ratio
        y = list(map(int, (data)))
        x = np.arange(len(y))
        pts = np.stack((x, y)).T
        cv2.line(sub_canvas, (0, absis), (w, absis), self.color_absis, self.thickness)
        cv2.polylines(sub_canvas, np.array([pts]), False, self.color_graph, self.thickness)

        # border
        border_size = 1
        sub_canvas[:, 0:border_size] = WHITE
        sub_canvas[:, -border_size:] = WHITE
        sub_canvas[0:border_size, :] = WHITE
        sub_canvas[-border_size:, :] = WHITE

        # union subcanvas and canvas
        canvas[pad_h:h - pad_h * text_box_size, pad_w:w - pad_w] = sub_canvas

        # statistic
        bottom_text = int(h * 0.8)
        left_w = int(w * 0.1)
        self.put_text(canvas, name, left_w, bottom_text)
        self.put_text(canvas, f"Current={round(self.data[-1], 4)}", left_w, int(h * 0.9))
        self.put_text(canvas, f"Max={round(data_max, 4)}", int(left_w * 6), bottom_text)
        self.put_text(canvas, f"Min={round(data_min, 4)}", left_w * 6, int(h * 0.9))

        return canvas


counter = 0
x = np.linspace(0, 500, 10000)
class GUI:
    def __init__(self, path, size):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.running = True
        width, height = int(self.cap.get(3)), int(self.cap.get(4))
        self.size = size

        self.graph = Graph(width, height)
        self.graph_classes = GraphClasses(width, height)
        self.graph_continuous = GraphContinuous(width, height)

    @lru_cache
    def resized_width_height(self, shape):
        h, w, _ = shape
        w, h = int(w * self.size[0]), int(h * self.size[1])
        return w, h

    def view(self, frame):
        classes = self.graph_classes.plot(np.random.random(7), (0.5, 0.5))
        plain_plot = self.graph.plot([], (0.5, 0.5))

        global counter, x
        y = np.sin(x[counter])
        counter += 1
        continuous_plot = self.graph_continuous.plot(y, (0.5, 0.5), "Class 1")

        frame = np.concatenate(
            (frame, np.concatenate((classes, continuous_plot), axis=1)),
            axis=0
        )

        # frame = np.concatenate()

        w, h = self.resized_width_height(frame.shape)
        frame = cv2.resize(frame, (w, h))
        cv2.imshow('Frame', frame)

    def controller(self):
        if cv2.waitKey(25) & 0xFF == ord('q'):
            self.running = False

    def run(self):
        assert self.cap.isOpened(), "Video is not opened"
        while self.cap.isOpened() and self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            self.view(frame) if ret else None
            self.controller()
            print(f"{time.time() - start_time}")
        self.cap.release()
        cv2.destroyAllWindows()
