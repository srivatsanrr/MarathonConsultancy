import argparse
import pytesseract as pts
import os
import cv2


args = argparse.ArgumentParser()

args.add_argument("-d", "--path", required=True,help="path to input folder")
args.add_argument("-m", "--model", required=False,help="path to input folder",nargs="?",const="./models/frozen_east_text_detection.pb")


vars = args.parse_args()

DETECTION_MODEL_PATH = vars.model
H,W = 800,1184

model = cv2.dnn.readNetFromTensorflow(DETECTION_MODEL_PATH)
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
    ]


def run_detection_for_each_image(path):
    
    for image_name in os.listdir(path):
        image = cv2.imread(os.path.join(path,image_name))
        

        cv2.imshow('Image',image)
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), 
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        model.setInput(blob)
        
        (scores, geometry) = model.forward(layerNames)

        cv2.waitKey(0)






if __name__ == '__main__':
    run_detection_for_each_image(vars.path)