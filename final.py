import cv2
import tim
from djitellopy import tello
from simple_facerec import SimpleFacerec
from cvzone.HandTrackingModule import HandDetector



"""
    The function  is used to send the drone commands to turn 
    according to the desired orientation.
    
     """



class Control:
    def __init__(self, tuner, final):
        self.tuner = tuner
        self.final = final
        self.limit = 0
        self.I = 0
        self.time = 0

    def orient(self, val):
        t = time.time() - self.pTime
        error = val - self.final
        P = self.tuner[0] * error
        D = (self.tuner[1] * (error) / t)

        result = P + self.I + D

        return result

class FaceDetector:
    """
    Face detction library made using MediaPipe Libraray 
     https://developers.google.com/mediapipe/solutions/vision/face_detector/python
        """

    def __init__(self, accuracy=0.5):
        
        self.accuracy = accuracy
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.accuracy)

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return the bbox info
        :param img: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings.
                 Bounding Box list.
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                if draw:
                    img = cv2.rectangle(img, bbox, (0, 255, 0), 2)

        return img, bboxs


"""
    Recognise faces in realtime using keras models which uses CNN to recognise the face;
     A simple model which uses less hyperparameters to tune which reduces time lag but tends
    to undertrain or over train.

    Classification Module
    Based on the following links:
    https://github.com/Shahnawax/HAR-CNN-Keras.git
    https://teachablemachine.withgoogle.com/
     """
faceRecon = SimpleFacerec()
faceRecon.load_encoding_images("images/")


faceDectector = FaceDetector()
detect = HandDetector(detectionCon=0.8, maxHands=1)

winHeight,winWidth=480,640


xAxis =Control([0.25,0.1],winWidth//2)
yAxis =Control([0.2,0.1],winHeight//2)
zAxis =Control([0.0009,0.0001],31000)


"""
    Creates the Drone Object using the official Tello API,
    The commands are send through the Drone through UDP connection
    The following lines of code :
    --> Connects to Drone
    --> Starts the Stream from Drone 
    --> Moves the drone up by 40 cm
     """

drone = tello.Tello()
drone.connect()
drone.streamoff()
drone.streamon()
drone.takeoff()
drone.move_up(40)



while True: 
    

     frame= drone.get_frame_read().frame
     frame =cv2.resize(frame,(640,480))
    
"""
FaceRecognition 
"""

    images, name = faceRecon.detect_known_faces(frame)
    for face_loc, name in zip(images, name):
        a, d, b, c = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(c, a - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (c, a), (d, b), (0, 0, 255), 3)
        name1 = name 


    frame,bboxs = faceDectector.findFaces(frame,draw = True)
    hands, img = detect.findHands(frame)

    ## If the face is detected properly
    ##src : https://www.youtube.com/watch?v=NZde8Xt78Iw

    if(name1 == "arsh"):

        """
        HandDetetction Code
        """
        if hands:
            
            hand1 = hands[0]
            fingerVal = detect.fingersUp(hand1)
      
        
            if(fingerVal == [1,1,1,1,1]):
                cv2.putText(frame,"5",(0,winHeight//2),cv2.FONT_HERSHEY_DUPLEX,8,(255,0,0),3) 
            elif(fingerVal == [0,1,1,1,1]):
                cv2.putText(frame,"4",(0,winHeight//2),cv2.FONT_HERSHEY_DUPLEX,8,(255,0,0),3) 
            elif(fingerVal == [0,1,1,1,0]):
                cv2.putText(frame,"3",(0,winHeight//2),cv2.FONT_HERSHEY_DUPLEX,8,(255,0,0),3) 
            elif(fingerVal == [0,1,1,0,0]):
                cv2.putText(frame,"2",(0,winHeight//2),cv2.FONT_HERSHEY_DUPLEX,8,(255,0,0),3) 
            elif(fingerVal == [0,1,0,0,0]):
                #cv2.putText(frame,"1",(0,winHeight),cv2.FONT_HERSHEY_DUPLEX,8,(255,0,0),3)  
                cv2.putText(frame,"Landing",(0,winHeight//2),cv2.FONT_HERSHEY_DUPLEX,4,(255,0,0),3) 
            elif(fingerVal == [0, 0, 0, 0, 0]):
                
                #drone.land()  

            

        if bboxs:

            cx,cy= bboxs[0]['center']
            a,b,w,h = bboxs[0]['bbox']
            k = (w*h)
            

            X = int(xAxis.orient(cx))
            Y = int(yAxis.orient(cy))
            Z = int(zAxis.orient(k))

        

            cv2.circle(frame,(cx,cy),5,(0,0,0),cv2.FILLED)

            cv2.line(frame,(winWidth//2,0),(winWidth//2,winHeight),(0,0,255),1)
            cv2.line(frame,(0,winHeight//2),(winWidth,winHeight//2),(0,0,255),1)

            error = winWidth//2 -cx

            cv2.line(frame,(winWidth//2,winHeight//2),(cx,cy),(0,0,255),1)

            drone.send_rc_control(0,-Z,-Y,X)
            name1 ="Not Arsh"
    
    

  

    
      
    cv2.imshow("Image",frame)
   
   
    if cv2.waitKey(5) & 0xFF == ord('q'):

        cv2.destroyAllWindows()
