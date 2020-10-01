#!/usr/bin/env python3
"""
Yolo Object Detection
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


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
        input_h = self.model.input.shape[2].value
        input_w = self.model.input.shape[1].value

        for outi in range(len(outputs)):
            net_outp = outputs[outi]
            grid_h, grid_w = net_outp.shape[:2]
            net_box = net_outp.shape[-2]
            net_class = net_outp.shape[-1] - 5
            anchors = self.anchors[outi]
            net_outp[..., :2] = self.sigmoid(net_outp[..., :2])
            net_outp[..., 4:] = self.sigmoid(net_outp[..., 4:])
            net_bx = net_outp[..., 4:]  # varible easier to use

            for r in range(net_outp.shape[0]):  # grid height/row
                for c in range(net_outp.shape[1]):  # grid width/col
                    for b in range(net_box):
                        x, y, w, h, = net_outp[r][c][b][:4]
                        x = (c + x)
                        ctr_x = x / grid_w  # center image width
                        y = (r + y)
                        ctr_y = y / grid_h  # center image height
                        w = (anchors[b][0] * np.exp(w))
                        im_width = w / input_w  # image width
                        h = (anchors[b][1] * np.exp(h))
                        im_height = h / input_h  # image height

                        # define scale
                        x_Box = (ctr_x - im_width/2) * image_w
                        y_Box = (ctr_y - im_height/2) * image_h
                        x_2Box = (ctr_x + im_width/2) * image_w
                        y_2Box = (ctr_x + im_width/2) * image_h

                        # can use BoundBox from plantar library
                        # ex. box = plantar.BoundBox(...)
                        net_bx[r, c, b, 0:4] = x_Box, y_Box, x_2Box, y_2Box
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: numpy.ndarray Shape(grid_height, grid_width, anchor_boxes, 4)
        box_confidences: numpy.ndarray
                        shape(grid_Height, grid_wisth, anchor_boxes, 1)
        box_class_probs: numpy.ndarray
                        shape(grid_height, grid_width, acnchor_boxes, classes)
        box_classes: numpy.ndarray shape (?,)
        box_scores: numpy.ndarray of shape (?) scores for  filtered_boxes
        Returns: tuple; (filtered_boxes, box_classes, box_scores)
                filtered_boxes: numpy.ndarray(?, 4) filtered bounding boxes
        """
        scores = []
        filtered_boxes = []
        box_classes = []
        box_scores = []
        for box_conf, box_class in zip(box_confidences, box_class_probs):
            scores.append(box_conf * box_class)

        for num in scores:
            # find highest box score for each section
            boxScore = num.max(axis=-1)
            boxScore = boxScore.flatten()
            box_classes.append(boxScore)

            # find index of highest numbers
            boxClassScore = np.argmax(scores, axis=-1)
            boxClassScore = boxClassScore.flatten()
            box_classes.append(boxClassScore)

            # include the axis=-1
            box_classes = np.concatenate(box_classes, axis=-1)
            box_scores = np.concatenate(box_scores, axis=-1)

            # creating the filtered bounding boxes
            for numBox in boxes:
                filtered_boxes.append(numBox.reshape(-1, 4))
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)

            mask = np.where(box_scores >= self.class_t)

            filtered_boxes = filtered_boxes[mask]
            box_classes = box_classes[mask]
            box_scores = box_scores[mask]

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Function performs non-max suppression on boundary boxes
        filtered_boxes: numpy.ndaray shape (?,4)
        box_classes: numpy.ndarray shape(?, )
                    containg all of the filtered_boxes
        box_scores: numpy.ndarray shape(?)
                    containing the box scores for each box
                    in filtered_boxes
        Returns: tuple (box_predictions, predicted_box_classes,
                        predicted_box_scores)
                box_predictions: numpy.ndarray shape (?, 4)
                                contains: predicted bounding boxes
                                ordered: by class and box score
                predicted_box_classes: numpy.ndarry shape (?, )
                                contains: class number for box_predictions
                                ordered: by class and box score
                predicted_box_scores: numpy.ndattay shape(?)
                                contains: box scores for box_predictions
                                ordered: by class and box score
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for boxClass in set(box_classes):
            idx = np.where(box_classes == boxClass)  # where they are the same

            # function arrays
            fb_i = filtered_boxes[idx]
            bc_i = box_classes[idx]
            bS_i = box_scores[idx]

            # coordinates of the bounding boxes
            x1 = fb_i[:, 0]
            y1 = fb_i[:, 1]
            x2 = fb_i[:, 2]
            y2 = fb_i[:, 3]

            # calculate area of the bounding boxes and sort
            union_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            sort_order = bS_i.argsort()[::-1]

            # loop remaining indexes
            pkd_idxs = []  # to hold list of picked indexes
            while len(sort_order) > 0:
                i_pos = sort_order[0]
                l_pos = sort_order[1:]
                pkd_idxs.append(i_pos)

            # find the coordinates of intersection
                xx1 = np.maximum(x1[i_pos], x1[l_pos])
                yy1 = np.maximum(y1[i_pos], y1[l_pos])
                xx2 = np.minimum(x2[i_pos], x2[l_pos])
                yy2 = np.minimum(y2[i_pos], y2[l_pos])

            # width and height of bounding box
            wb = np.maximum(0, xx2 - xx1 + 1)
            hb = np.maximum(0, yy2 - yy1 + 1)

            # overlap ratio betw bounding box
            interSect = (hb * wb)
            overlap = union_area[i_pos] + union_area[l_pos] - interSect
            iou = interSect / overlap

            # below Threshold
            below_Thresh = np.where(iou <= self.nms_t)[0]
            sort_order = sort_order[below_Thresh + 1]

            pkd = np.array(pkd_idxs)  # array of piked indexes

            # append picked to function arrays
            box_predictions.append(pkd)
            predicted_box_classes.append(pkd)
            predicted_box_scores.append(pkd)

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        load images
        """
        images = []
        image_paths = glob.glob(folder_path + "/*")

        for m in image_paths:
            image = cv2.imread(m)
            images.append(image)

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        function resizes and rescales images
        """
        input_h = self.model.input.shape[2].value
        input_w = self.model.input.shape[1].value
        nSize_images = []
        im_shape = []

        for m in images:
            image = cv2.resize(m, (input_w, input_h),
                               interpolation=cv2.INTER_CUBIC)
            image = image.astype(float) / 255  # colors
            nSize_images.append(image)
        pimages = np.stack(nSize_images, axis=0)
        im_shape = [m.shape[:2] for m in images]  # apply shape to images
        image_shapes = np.stack(im_shape, axis=0)
        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        function displays image with: boundary boxes
                                      class names
                                      box scores
        image: numpy.ndarray containing unprocessed image
        boxes: numpy.ndarray containing boundary boxes for image
        box_classes: numpy.ndarray containing class indices for each box
        box_scores: numpy.ndarray containing box scores for each box
        file_name: file path where original image stored
        """
        orig_image = image
        for idx, box in enumerate(boxes):
            # bounding boxes
            start_x = int(box[0])
            start_y = int(box[1])
            end_x = int(box[2])
            end_y = int(box[3])

            # color items
            textColor = (0, 0, 255)
            lineColor = (255, 0, 0)

            # box charateristics
            orig_image = cv2.rectangle(orig_image,
                                       (start_x, start_y), (end_x, end_y),
                                       lineColor, thickness=2)
            # text chracteristics
            orig_image = cv2.putText(orig_image,
                                     self.class_names[box_classes[idx]]
                                     + " " + "{:.2f}".format(box_scores[idx]),
                                     (start_x, (end_y) - 5),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5, textColor,
                                     thickness=1,
                                     lineType=cv2.LINE_AA,
                                     bottomLeftOrigin=False)
            cv2.imshow(file_name, image)
            key = cv2.waitKey(0)
            if key == ('s'):
                if not os.path.exists('detections'):
                    os.makedirs('detections')
                os.chdir('detections')
                cv2.imwrite(file_name, image)
                os.chdir('../')
            cv2.destroyAllWindows()
