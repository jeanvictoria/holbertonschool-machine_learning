#!/usr/bin/env python3
"""Yolo algorithm construction"""

import tensorflow.keras as K
import numpy as np


class Yolo():
    """class Yolo that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ initialization of the variables
        Arg:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path with the list of class names used,
                          listed in order of index, can be found
            class_t: float, box score threshold for the init filtering step
            nms_t: float representing the IOU threshold for non-max suppression
            anchors: np.ndarray.shape (outputs, anchor_boxes, 2)
                     has all anchor boxes:
                     outputs: number of outputs (predictions)
                     anchor_boxes: num of anchor boxes used for each prediction
                     2 => [anchor_box_width, anchor_box_height]
        """
        class_names = []
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as classes_file:
            class_names = classes_file.readlines()
        class_names = [x.strip() for x in class_names]

        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ process the outputs
        Arg:
            outputs: list of numpy.ndarrays containing the predictions:
                Each output will have the shape
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
                     grid_height & grid_width =>
                     the height and width of the grid used for the output
                     anchor_boxes => the number of anchor boxes used
                     4 => (t_x, t_y, t_w, t_h)
                     1 => box_confidence
                     classes => class probabilities for all classes
            image_size: numpy.ndarray image’s original size [height, width]
        Returns: tuple of (boxes, box_confidences, box_class_probs):
                 boxes: np shape (grid_height, grid_width, anchor_boxes, 4)
                 box_confidences: np.shape (grid_height, grid_width,
                    anchor_boxes, 1) has box confidences for each output
                 box_class_probs:(grid_height, grid_width, anchor_boxes,
                    classes) box’s class probabilities for each output
        """
        boxes = []
        box_confidence = []
        box_class_probs = []

        for i in range(len(outputs)):
            grid_h, grid_w, nb_box, _ = outputs[i].shape

            box_conf = 1 / (1 + np.exp(-(outputs[i][:, :, :, 4:5])))
            box_confidence.append(box_conf)
            box_prob = 1 / (1 + np.exp(-(outputs[i][:, :, :, 5:])))
            box_class_probs.append(box_prob)

            box_xy = 1 / (1 + np.exp(-(outputs[i][:, :, :, :2])))
            box_wh = np.exp(outputs[i][:, :, :, 2:4])
            anchors_tensor = self.anchors.reshape(1, 1,
                                                  self.anchors.shape[0],
                                                  nb_box, 2)
            box_wh = box_wh * anchors_tensor[:, :, i, :, :]

            col = np.tile(np.arange(0, grid_w),
                          grid_h).reshape(grid_h, grid_w)
            row = np.tile(np.arange(0, grid_h),
                          grid_w).reshape(grid_w, grid_h).T
            col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
            row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
            grid = np.concatenate((col, row), axis=3)

            box_xy += grid
            box_xy /= (grid_w, grid_h)
            input_h = self.model.input.shape[2].value
            input_w = self.model.input.shape[1].value
            box_wh /= (input_w, input_h)
            box_xy -= (box_wh / 2)
            box_xy1 = box_xy
            box_xy2 = box_xy1 + box_wh
            box = np.concatenate((box_xy1, box_xy2), axis=-1)

            box[..., 0] *= image_size[1]
            box[..., 2] *= image_size[1]
            box[..., 1] *= image_size[0]
            box[..., 3] *= image_size[0]

            boxes.append(box)

        return ((boxes, box_confidence, box_class_probs))

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ filtering the boxes with object threshold
        Arg:
            boxes: np shape (grid_height, grid_width, anchor_boxes, 4)
                   with the process outputs boxes
            box_confidences: shape (grid_height, grid_width, anchor_boxes, 1)
                   has all the processed box confidences
            box_class_probs: (grid_height, grid_width, anchor_boxes, classes)
                   has all the processed box class probabilities
        Returns tuple of (filtered_boxes, box_classes, box_scores)
            filtered_boxes: np array shape (?, 4) all filtered bounding boxes
            box_classes: np array shape (?,) class number that each box in
                         filtered_boxes predicts
            box_scores: np array of shape (?) the box scores for each box
                        in filtered_boxes
        """
        box_score = []
        for confis, probs in zip(box_confidences, box_class_probs):
            score = confis * probs
            box_score.append(score)

        box_class_scores = [score.max(axis=-1) for score in box_score]
        box_score_list = [box.reshape(-1) for box in box_class_scores]
        box_scores_conca = np.concatenate(box_score_list, axis=-1)

        filt = np.where(box_scores_conca >= self.class_t)

        box_scores = box_scores_conca[filt]

        boxes = [box.reshape(-1, 4) for box in boxes]
        boxes_conca = np.concatenate(boxes, axis=0)
        filtered_boxes = boxes_conca[filt]

        box_class = [score.argmax(axis=-1) for score in box_score]
        box_class_list = [box.reshape(-1) for box in box_class]
        box_class_conca = np.concatenate(box_class_list, axis=-1)
        box_classes = box_class_conca[filt]

        return (filtered_boxes, box_classes, box_scores)

    def keeped_boxes_iou(self, filtered_boxes, box_scores):
        """ selecting the boxes to keep by its score index
            Arg:
                - filtered_boxes: np.ndarray of boxes filtered
                - box_scores: np.ndarray of scores filteres
            Return: np.ndarray with the indexes of the box_scores to keep
        """
        x1 = filtered_boxes[:, 0]
        y1 = filtered_boxes[:, 1]
        x2 = filtered_boxes[:, 2]
        y2 = filtered_boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        order = box_scores.argsort()[::-1]

        keep = []
        # while there are boxes:
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # find the intersection values
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # find the weight and height max
            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1
            # IOU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_t)[0]
            # filtering the list, quit values less than nms_t variable
            order = order[inds + 1]

        keep = np.array(keep)
        return(keep)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ Non-max suppression to take the best box option for obj detection
        Arg:
            filtered_boxes: np shape (?, 4) containing all of the
                            filtered bounding boxes.
            box_classes: np shape (?,) containing the class number
                         that the filtered_boxes predicts
            box_scores: np shape (?) box scores for each box in filtered_boxes
        Returns:(box_predic, predic_box_classes, predic_box_scores)
            box_predictions: np shape (?, 4) all of the predicted
                             bounding boxes ordered by class and box score
            predicted_box_classes: np shape (?,) with class number for
                                   box_predictions
            predicted_box_scores: np shape (?) with the box scores for
                                  box_predictions
        """
        nboxes, nclasses, nscores = [], [], []
        for c in set(box_classes):
            inds = np.where(box_classes == c)
            b = filtered_boxes[inds]
            c = box_classes[inds]
            s = box_scores[inds]

            keep = self.keeped_boxes_iou(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        boxes_predic = np.concatenate(nboxes)
        classes_predic = np.concatenate(nclasses)
        scores_predic = np.concatenate(nscores)

        return (boxes_predic, classes_predic, scores_predic)
