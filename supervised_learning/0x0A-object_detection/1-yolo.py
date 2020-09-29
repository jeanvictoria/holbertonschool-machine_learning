#!/usr/bin/env python3
"""
Yolo Object Detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """Class constructor"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to Darknet Keras model
        classes_path: path to list of class names for Darnet model
                      list in order of index
        class_t: float = box score threshold for inital filtering
        nms_t: float for IoU threshold for non-max suppression
        anchors: numpy.darray shape(outputs, anchor_boxes, 2)
                 contains all anchor boxes
        ouputs: num Darknet Model outputs
        anchor_boxes: num anchor boxes used for each prediction
        2 -> [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classFile:
            self.class_names = [line.strip() for line in classFile]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """sigmoid function"""
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Process Outputs Darknet
        outputs: list of numpy.ndarray s;
                contains Darknet predictions from image
                shape(grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            grid_height: height of grid used for out
            grid_width: width of grid used for out
            anchor_boxes: num of anchor boxes used
            4: (t_x, t_y, t_w, t_h)
            1: box_confidence
        classes: classes probabilites for all classes
        image_size: numpy.ndarray containing image’s original size
            [image_height, image_width]
        Returns: (boxes, box_confidences, box_class_probs)
            boxes: list numpy.ndarrays
                shape(grid_height, grid_width, anchor_boxes, 4)
                containing the processed boundary boxes for each out
            box_class_probs: list numpy.ndarrays
                shape(grid_height, grid_width, anchor_boxes, classes)
                containing box’s class probabilities for each
            box_confidences: list of numpy.ndarrays
                shape (grid_height, grid_width, anchor_boxes, 1)
                 containing box confidences for each out
        """

        boxes = []
        box_class_probs = []
        box_confidences = []
        image_h = image_size[0]
        image_w = image_size[1]
        input_h = self.model.input.shape[1].value
        input_w = self.model.input.shape[2].value

        for outi in range(len(outputs)):
            net_outp = outputs[outi]
            grid_h, grid_w = net_outp.shape[:2]
            net_box = net_outp.shape[-2]
            net_class = net_outp.shape[-1] - 5
            anchors = self.anchors[outi]
            net_outp[..., :2] = self.sigmoid(net_outp[..., :2])
            net_outp[..., 4:] = self.sigmoid(net_outp[..., 4:])
            net_bx = net_outp[..., :4]  # varible easier to use

            for r in range(grid_h):
                for c in range(grid_w):
                    for b in range(net_box):
                        y, x, w, h, = net_bx[r, c, b, :4]
                        # center image height and width
                        x = (c + x)
                        ctr_x = x / grid_w
                        y = (r + y)
                        ctr_y = y / grid_h
                        # image height and width
                        w = (anchors[b][0] * np.exp(w))
                        im_width = w / input_w
                        h = (anchors[b][1] * np.exp(h))
                        im_height = h / input_h

                        # define scale
                        x_Box = (ctr_x - im_width/2) * image_w
                        y_Box = (ctr_y - im_height/2) * image_h
                        x_2Box = (ctr_x + im_width/2) * image_w
                        y_2Box = (ctr_y + im_width/2) * image_h

                        # can use BoundBox from plantar library
                        # ex. box = plantar.BoundBox(...)
                        net_bx[r, c, b, 0:4] = y_Box, x_Box, y_2Box, x_2Box
                        boxes.append(net_bx)  # boxes contain scale

            # output confidences
            # box_conf = [self.sigmoid(net_outp[..., 4:5])]
            box_conf = net_outp[..., 4:5]
            box_confidences.append(box_conf)

            # output probabilities
            # box_class_p = [self.sigmoid(net_outp[..., 5:])]
            box_class_p = net_outp[..., 5:]
            box_class_probs.append(box_class_p)

        return (boxes, box_confidences, box_class_probs)
