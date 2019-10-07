import sys
import cv2
import numpy as np
import os.path
import glob
import argparse


conf_threshold = 0.25                           # Confidence threshold
nms_threshold = 0.4                             # Non-maximum suppression threshold
input_size = (512, 512)                         #Size of input images
window_name = "Detection"
window_size = (1920, 1080)


# Load names of classes
def load_class_names(labels_file):
    classes = None
    with open(labels_file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    return classes


# Get the names of the output layers
def get_output_names(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def draw_bbox(image, classes, class_id, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (class_id < len(classes))
        label = '%s:%s' % (classes[class_id], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                  (0, 0, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


# Remove the bounding boxes with low confidence using non-maxima suppression
def post_process(image, outs, classes):
    frame_height = image.shape[0]
    frame_width = image.shape[1]
    class_ids = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_bbox(image, classes, class_ids[i], confidences[i], left, top, left + width, top + height)


# Functions to handle user interactions
def manage_interaction(img, img_file):
    cv2.imshow(window_name, img)
    k = cv2.waitKey()

    if k == ord('s'):   # Save result image to disk
        folder = os.path.dirname(img_file) + '/'
        filename = os.path.basename(img_file)
        filename, file_extension = os.path.splitext(filename)
        save_path = folder + filename + "_detected" + file_extension
        cv2.imwrite(save_path, img)
        print("Processed image saved: ", save_path)

    return k


# Detection function
def detect(image_file, config_file, weights_file, classes_file):
    if not os.path.isfile(image_file):
        print("Input image file ", image_file, " doesn't exist")
        sys.exit(1)
    else:
        print("Starting detection on image ", image_file, "...")

    img = cv2.imread(image_file)
    classes = load_class_names(classes_file)
    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(img, 1 / 255, input_size, [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    out_layers = get_output_names(net)
    outs = net.forward(out_layers)

    # Remove the bounding boxes with low confidence
    post_process(img, outs, classes)

    # Print efficiency information.
    # The function getPerfProfile returns the overall time for inference and the timings for each of the layers
    t, _ = net.getPerfProfile()
    time_txt = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    print(time_txt)
    return img


# Launch detection for all images of the given folder
def detect_folder(image_folder, config_file, weights_file, classes_file):
    images = glob.glob(image_folder + "/*.jpg")
    for filename in images:
        img = detect(filename, config_file, weights_file, classes_file)
        k = manage_interaction(img, filename)

        if k == 27:  # Esc key to stop
            print("Detection aborted by user")
            break


# Test only
def test_detection():
    classes_file = "YoloV3/classes.names"
    config_file = "YoloV3/Training1/yolov3_inference.cfg"
    weights_file = "YoloV3/Training1/yolov3_best.weights"
    image_folder = "YoloV3/Dataset"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_size)
    detect_folder(image_folder, config_file, weights_file, classes_file)


if __name__ == '__main__':
    # test_detection()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_path",
        help="Image path or folder path containing images on which detection will be applied"
    )

    parser.add_argument(
        "config_file",
        help="YOLO config file"
    )

    parser.add_argument(
        "weights_file",
        help="YOLO weights file"
    )

    parser.add_argument(
        "class_file",
        help="File containing class label(s)"
    )

    print("----- Information -----\r")
    print("Press 's' key to save current image detection\r")
    print("Press any key to detect next image\r")
    print("Press 'Esc' key to abort\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_size)
    args = parser.parse_args()

    if os.path.isfile(args.source_path):
        img = detect(args.source_path, args.config_file, args.weights_file, args.class_file)
        manage_interaction(img, args.source_path)
    elif os.path.isdir(args.source_path):
        detect_folder(args.source_path, args.config_file, args.weights_file, args.class_file)
