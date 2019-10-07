# YoloDetection
Implementation of YOLO DNN inference with OpenCV

We give a Python script to launch inference (detection):

```shell
python detection.py "SourceImage" "ConfigFile" "WeightsFile" "ClassFile"
```

**SourceImage (mandatory)**: path of the image or folder of images on which detection will be done.  
**ConfigFile (mandatory)**: path of the YOLO configuration file (.cfg).  
**WeightsFile (mandatory)**: path of the YOLO trained model (.weights).  
**ClassFile (mandatory)**: path to the text file where are listed all class labels.

Once the detection is done, the result image is shown with bounding boxes around detected objects.
Here are ths list of possible user interaction:
1. Press *'s'* key to save result image into the current folder.
2. Press *'Esc'* key to end process.
3. Press any other key to process next image
