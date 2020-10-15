#!/usr/bin/env python3

"""Yolo class v3"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo v3"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor method
            - model_path is the path to where a Darknet Keras model is stored
            - classes_path is the path to where the list of class names used
             for the Darknet model, listed in order of index, can be found
            - class_t is a float representing the box score threshold
             for the initial filtering step
            - nms_t is a float representing the IOU threshold for
             non-max suppression
            - anchors is a numpy.ndarray of shape (outputs, anchor_boxes,
             2) containing all of the anchor boxes:
            - outputs is the number of outputs (predictions) made by
             the Darknet model
            - anchor_boxes is the number of anchor boxes used for each
             prediction
                2 => [anchor_box_width, anchor_box_height]"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = [line.rstrip() for line in file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, z):
        """Calculates sigmoid function"""
        return (1 / (1 + np.exp(-z)))

    def process_outputs(self, outputs, image_size):
        """
        Process outputs method
        """

        img_height = image_size[0]
        img_width = image_size[1]

        boxes = [output[..., 0:4] for output in outputs]
        box_confidences = [self.sigmoid(
            output[..., 4, np.newaxis]) for output in outputs]
        box_class_probs = [self.sigmoid(output[..., 5:]) for output in outputs]

        for i, box in enumerate(boxes):

            grid_height, grid_width, anchor_boxes, _ = box.shape

            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

            idx_y = np.arange(grid_height)
            idx_y = idx_y.reshape(grid_height, 1, 1)
            idx_x = np.arange(grid_width)
            idx_x = idx_x.reshape(1, grid_width, 1)

            C_x = c + idx_x
            C_y = c + idx_y

            centerX = box[..., 0]
            centerY = box[..., 1]
            width = box[..., 2]
            height = box[..., 3]

            bx = (self.sigmoid(centerX) + C_x) / grid_width
            by = (self.sigmoid(centerY) + C_y) / grid_height

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = (np.exp(width) * pw) / self.model.input.shape[1].value
            bh = (np.exp(height) * ph) / self.model.input.shape[2].value

            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh
            box[..., 0] = x1 * img_width
            box[..., 1] = y1 * img_height
            box[..., 2] = x2 * img_width
            box[..., 3] = y2 * img_height

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes method"""
        obj_thresh = self.class_t

        box_scores_full = []
        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            box_scores_full.append(box_conf * box_class_prob)

        box_scores_list = [score.max(axis=3) for score in box_scores_full]
        box_scores_list = [score.reshape(-1) for score in box_scores_list]
        box_scores = np.concatenate(box_scores_list)

        index_to_delete = np.where(box_scores < obj_thresh)

        box_scores = np.delete(box_scores, index_to_delete)

        box_classes_list = [box.argmax(axis=3) for box in box_scores_full]
        box_classes_list = [box.reshape(-1) for box in box_classes_list]
        box_classes = np.concatenate(box_classes_list)
        box_classes = np.delete(box_classes, index_to_delete)

        boxes_list = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(boxes_list, axis=0)
        filtered_boxes = np.delete(boxes, index_to_delete, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non Max Supression"""

        index = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in index])
        predicted_box_classes = np.array([box_classes[i] for i in index])
        predicted_box_scores = np.array([box_scores[i] for i in index])

        _, class_counts = np.unique(predicted_box_classes, return_counts=True)

        i = 0
        accumulated_count = 0

        for class_count in class_counts:
            while i < accumulated_count + class_count:
                j = i + 1
                while j < accumulated_count + class_count:

                    box1 = box_predictions[i]
                    box2 = box_predictions[j]
                    xi1 = max(box1[0], box2[0])
                    yi1 = max(box1[1], box2[1])
                    xi2 = min(box1[2], box2[2])
                    yi2 = min(box1[3], box2[3])
                    inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

                    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
                    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
                    union_area = box1_area + box2_area - inter_area

                    iou = inter_area / union_area

                    if iou > self.nms_t:
                        box_predictions = np.delete(box_predictions, j,
                                                    axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         j, axis=0)
                        predicted_box_classes = (np.delete
                                                 (predicted_box_classes,
                                                  j, axis=0))
                        class_count -= 1
                    else:
                        j += 1
                i += 1
            accumulated_count += class_count

        return box_predictions, predicted_box_classes, predicted_box_scores
