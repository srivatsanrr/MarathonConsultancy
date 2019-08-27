import argparse
import pytesseract as pts
import os
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


# args = argparse.ArgumentParser()

# args.add_argument("-d", "--path", required=True,help="path to input folder")
# args.add_argument("-m", "--model", required=False,help="path to input folder",nargs="?",const="./models/frozen_east_text_detection.pb")
# args.add_argument("-c","--min_confidence",required=False,help="set non-max supression confidence",nargs="?",const=0.5,type=float)
# # args.add_argument("-p","--padding",required=True,)

# vars = args.parse_args()

H,W = None,None
layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
            ]


class OcrApi():



    def __init__(self,
        path,
        model_path="./models/frozen_east_text_detection.pb",
        min_confidence=0.7,
        padding=None,
        height=800,
        width=1184,):
        
        self.path = path
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.padding = padding
        self.height = height
        self.width = width
        
        self._model = cv2.dnn.readNetFromTensorflow(self.model_path)
        

    def run_detection_for_each_image(self,path):
        H,W = self.height,self.width
        for image_name in os.listdir(path):
            image = cv2.imread(os.path.join(path,image_name))
            
            image_copy = image.copy()

            origH,origW = image.shape[0:2] 

            blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), 
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
            self._model.setInput(blob)

            rW = origW/W
            rH = origH/H





            
            (scores, geometry) = self._model.forward(layerNames)

            rects, confidences = self._decode_predictions(scores,geometry)

            supressed_boxes = non_max_suppression(np.array(rects),probs=confidences)



            for (startX,startY,endX,endY) in supressed_boxes:
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                
                cv2.rectangle(image_copy, (startX, startY), (endX, endY), (0,0,255), 2)

            cv2.imshow('Image',image_copy)



            cv2.waitKey(0)

    def _decode_predictions(self,scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # etract the scores (probabilites), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0,0,y]
            xData0 = geometry[0,0,y]
            xData1 = geometry[0,1,y]
            xData2 = geometry[0,2,y]
            xData3 = geometry[0,3,y]
            anglesData = geometry[0,4,y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability
                # ignore it
                if scoresData[x] < self.min_confidence:
                    continue

                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x*4.0, y*4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)- coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin*xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos*xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)





if __name__ == '__main__':
    api = OcrApi(path="./test_images/")
    api.run_detection_for_each_image(api.path)
