from ultralytics import YOLO
import cv2
import os
import imutils
import numpy as np
import sys
import re

model=YOLO('Model/plateSentry.pt')

###############OCR MODEL##################


import easyocr

reader = easyocr.Reader(['en'])

def ocr(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]
    result = reader.readtext(thresholded_image)
    text = ''.join([res[-2] for res in result])
    accuracy=[]
    for i in  range(len(result)):
        accuracy.append(result[i][-1])
    accuracy=round(np.mean(accuracy)*100,4)
    data="Text : "+text +"    Accuracy : " + str(accuracy)
    return data
    

############### LIVE FEED ################


def livefeed():
    # Create a VideoCapture object and set the camera source
    cap = cv2.VideoCapture(0)

    # Loop through the frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

################IMAGE####################

def image_detection(img):
    # Run YOLOv8 inference on the image
    if isinstance(img, str):
        img = cv2.imread(img)
    results = model(img)
    # Visualize the results on the image
    annotated_image = results[0].plot()
    return annotated_image

###############VIDEO#####################
def video_detection(video):
    #if video is path to video
    if isinstance(video, str):
        cap= cv2.VideoCapture(video)
    else:
        cap = video
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the output video path and name
    base_path =os.getcwd()
    output_path = os.path.join(base_path, "static", "predict", "output.mp4")
    
    # Define the codec for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create the video writer object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Loop through the frames
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            # Apply your model to the frame
            infer = model(frame)
            annotated = infer[0].plot()
            
            # Write the annotated frame to the output video
            out.write(annotated)
            
            # Display the resulting frame
            cv2.imshow('Frame', annotated)
            
            # Press 'q' on the keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Release the video capture and writer objects
    cap.release()
    out.release()
    
    # Close all the frames
    cv2.destroyAllWindows()
    return output_path
##################Plate recognition using computer vision################



def process_image(path):
    def load_image(path):
        try:
            image=cv2.imread(path)
            return image
        except Exception as e:
            print("Path Error!")
            sys.exit()


    def preprocess(image):
        try:
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            filter=cv2.bilateralFilter(gray,10,17,17)
            return (gray,filter)
        except Exception as e:
            print("Image not found !")
            sys.exit()

    def find_edges(filter):
        try:
            edges=cv2.Canny(filter,threshold1=30,threshold2=200)    
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
            return edges
        except Exception as e:
            print(f"Error finding edges !")
            sys.exit()

    def localize(edges):
        try:
            coordinates=cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours=imutils.grab_contours(coordinates)
            contours=sorted(contours,key=cv2.contourArea, reverse=True)[:10]
            loc=None
            for point in contours:
                perimeter = cv2.arcLength(point, True)
                approx = cv2.approxPolyDP(point, 0.018 * perimeter, True)

                if len(approx)==4:
                    loc=approx
                    break
            return loc
        except Exception as e:
            print(f"Error localizing contours !")
            sys.exit()

    def mask(img,gray,loc):
        try:
            mask=np.zeros(gray.shape[0:2],np.uint8)
            plate=cv2.drawContours(mask,[loc],0,255,-1)
            plate=cv2.bitwise_and(img,img,mask=mask)
            (x,y)=np.where(mask==255)
            (x1,y1)=(np.min(x),np.min(y))
            (x2,y2)=(np.max(x),np.max(y))
            final_plate=img[x1:x2+1,y1:y2+1]
            return final_plate
        except Exception as e:
            print(f"Error masking image !")
            sys.exit()

    try:
        image = load_image(path)
        gray_image, filter = preprocess(image)
        edges = find_edges(filter)
        location = localize(edges)
        final = mask(image, gray_image, location)
        final=cv2.cvtColor(final,cv2.COLOR_BGR2RGB)
        text,accuracy=ocr(final)
        text=text.upper().replace('IND','')
        print(f"\nNumberplate - {re.sub(r'[^a-zA-Z0-9]', '', text)}")
        print(f"Accuracy - {np.mean(accuracy)*100}%")
        data="Text : "+text +"    Accuracy : " + str(accuracy)
        return data
    except Exception as e:
        print(f"Error processing image !")
        sys.exit()

##########################################################################
