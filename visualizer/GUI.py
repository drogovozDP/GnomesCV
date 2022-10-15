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
        data = data * graph_ratio
        data += absis
        data = sub_h - data
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


class GraphHist(Graph):
    def __init__(self, *args):
        super(GraphHist, self).__init__(*args)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_text = GRAPH_HIST["text"]
        self.color_rect = GRAPH_HIST["rect"]
        self.thickness = 2
        self.font_scale = 1
        self.numbers_panel = None

    def plot(self, data, ratio, *args):
        canvas = super(GraphHist, self).plot(data, ratio)
        sub_canvas, sub_w, sub_h, pad_w, pad_h, text_box_size = self.sub_plot(canvas, ratio)
        w, h = self.resize(ratio)
        sub_canvas += WHITE

        # calculate data
        hist = np.zeros(33)
        for x in data:
            idx = int(x // 10)
            idx = -1 if idx >= 32 else idx
            hist[idx] += 1

        max_hist = hist.max()
        hist /= max_hist if max_hist != 0 else 1

        # draw bins
        step_w = sub_w // len(hist)
        for i, col in enumerate(hist):
            cv2.rectangle(sub_canvas, (step_w * i, sub_h), (int(step_w * (i + 0.5)), int(sub_h * (1 - col))), self.color_rect, -1)
            cv2.rectangle(sub_canvas, (step_w * i, sub_h), (int(step_w * (i + 0.5)), int(sub_h * (1 - col))), RED, self.thickness)

        # border
        sub_canvas[:, :3] = RED
        sub_canvas[:, -3:] = RED
        sub_canvas[:3, :] = RED
        sub_canvas[-3:, :] = RED

        # union subcanvas and canvas
        canvas[pad_h:h - pad_h * text_box_size, pad_w:w - pad_w] = sub_canvas

        # set numbers
        if self.numbers_panel is None:
            # text_bottom = int(h * 0.9)
            numbers_panel = np.zeros((sub_w, 100, 3))
            step_h = sub_w // 33
            for i in range(32):
                num = f"{(i + 1) * 10}"
                num = " " * (4 - len(num)) + num
                self.put_text(numbers_panel, num, 12, step_h * (i + 1))
            self.numbers_panel = np.rot90(numbers_panel)

        place_h = pad_h + sub_h
        canvas[place_h:place_h + 100, pad_w:w - pad_w] = self.numbers_panel
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
        self.out = None

        self.graph = Graph(width, height)
        self.graph_classes = GraphClasses(width, height)
        self.graph_hist = GraphHist(width * 4, height)
        self.graph_continuous_1 = GraphContinuous(width, height)
        self.graph_continuous_2 = GraphContinuous(width, height)
        self.graph_continuous_3 = GraphContinuous(width, height)
        self.graph_continuous_4 = GraphContinuous(width, height)
        self.graph_continuous_5 = GraphContinuous(width, height)
        self.graph_continuous_6 = GraphContinuous(width, height)
        self.graph_continuous_7 = GraphContinuous(width, height)

    @lru_cache
    def resized_width_height(self, shape):
        h, w, _ = shape
        w, h = int(w * self.size[0]), int(h * self.size[1])
        return w, h

    def view(self, frame):
        classes = self.graph_classes.plot(np.random.random(7), (0.5, 0.5))
        plain_plot = self.graph.plot([], (0.5, 0.5))

        # here we will predict frame and calculate all stuff
        global counter, x
        cur_x = x[counter]
        counter += 1
        continuous_plot = self.graph_continuous_1.plot(np.sin(cur_x), (0.5, 0.5), "Class 1 (sin)")

        frame = np.concatenate(
            (frame, np.concatenate((classes, continuous_plot), axis=1)),
            axis=0
        )

        frame = np.concatenate(
            (frame, np.concatenate((
                self.graph_continuous_2.plot(np.cos(cur_x), (0.5, 0.5), "Class 2 (cos)"),
                self.graph_continuous_3.plot(np.tan(cur_x), (0.5, 0.5), "Class 3 (tan)"),
                self.graph_continuous_4.plot(np.random.random() * 2 - 1, (0.5, 0.5), "Class 4 (uniform)")
            ), axis=0)),
            axis=1
        )

        frame = np.concatenate(
            (frame, np.concatenate((
                self.graph_continuous_5.plot(np.random.normal() * 2 - 1, (0.5, 0.5), "Class 5 (normal)"),
                self.graph_continuous_6.plot(cur_x, (0.5, 0.5), "Class 6 (x)"),
                self.graph_continuous_7.plot(cur_x**(1/2), (0.5, 0.5), "Class 7 (x^(1/2))"),
            ), axis=0)),
            axis=1
        )

        frame = np.concatenate((frame, self.graph_hist.plot(np.random.rand(1000) * 340, (0.5, 0.5))), axis=0)

        w, h = self.resized_width_height(frame.shape)
        frame = cv2.resize(frame, (w, h))
        cv2.imshow('graphs', frame)
        self.out.write(frame) if self.out is not None else None

    def controller(self):
        if cv2.waitKey(25) & 0xFF == ord('q'):
            self.switch_off()

    def switch_off(self):
        self.running = False

    def run(self, save_video):
        if save_video:
            self.out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                  (1536, 864))
        assert self.cap.isOpened(), "Video is not opened"
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            self.view(frame) if ret else self.switch_off()
            self.controller()
            print(f"{time.time() - start_time}")
        self.cap.release()
        cv2.destroyAllWindows()
