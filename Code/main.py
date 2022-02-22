import os
import cv2 as cv
import sys
import argparse

input_video = 0

def parser():
   parser = argparse.ArgumentParser(description="Pretrain Age gender")
   parser.add_argument('--input',type=str, default= 0)
   
   return parser.parse_args()

def str2int(video_path):
   try:
      return int(video_path)
   except ValueError:
      return video_path

def facebox(faceNet,frame):
   #print("frame",frame)
   frame_w = frame.shape[1]
   #print("frame_w",frame_w)
   frame_h = frame.shape[0]
   #print("frame_h",frame_h)
   blob = cv.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB = False)
   faceNet.setInput(blob)
   detection =  faceNet.forward()
   bboxs = []
   for i in range(detection.shape[2]):
      confidence = detection[0,0,i,2]
      if confidence > 0.9:
         x1 = int(detection[0,0,i,3]*frame_w)
         y1 = int(detection[0,0,i,4]*frame_h)
         x2 = int(detection[0,0,i,5]*frame_w)
         y2 = int(detection[0,0,i,6]*frame_h)
         bboxs.append([x1,y1,x2,y2])
         cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
         
   return frame,bboxs

def set_saved_video(input_video, output_video, size):
   fourcc = cv.VideoWriter_fourcc(*"MP4V")
   fps = int(input_video.get(cv.CAP_PROP_FPS))
   print("fps: ",fps)
   video = cv.VideoWriter(output_video,fourcc,fps,size)
   return video

# added the face model and face proto for face detection 
# pretrained TensorFlow
facemodel = "model_config\opencv_face_detector_uint8.pb"
faceproto = "model_config\opencv_face_detector.pbtxt"

# Pretrained Caffe Model
#add the age model and age proto
agemodel= r"model_config\age_net.caffemodel"
ageproto = r"model_config\age_deploy.prototxt"

# path the gender model and proto for gender prediction
gendermodel = "model_config\gender_net.caffemodel"
genderproto = "model_config\gender_deploy.prototxt" 

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

#using cv2 read model
faceNet = cv.dnn.readNet(facemodel,faceproto)
ageNet = cv.dnn.readNet(agemodel,ageproto)
genderNet = cv.dnn.readNet(gendermodel,genderproto)


video = cv.VideoCapture(input_video)
if (video.isOpened() == False):
   print("Error reading video file")
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))


def main():
   result = set_saved_video(video,"pretrain_Tensor_webcam1.mp4",(width,height))
   while True:
      ret,frame = video.read()
      if ret == True:
         frame, bboxs = facebox(faceNet,frame)
         print("len(bboxs): ",len(bboxs))
         print("bboxs:",bboxs)
         if len(bboxs) == 0:
            pass
         else:
            for bbox in bboxs:
               #crop face from frame detection
               face = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
               print("face: ",face)
               #face = cv.resize(face,(224,224))
               #print("face: ",face)
               blob = cv.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB = False)
               genderNet.setInput(blob)
               gender_predict = genderNet.forward()
               gender = genderList[gender_predict[0].argmax()]
               
               ageNet.setInput(blob)
               agePre =  ageNet.forward()
               age = ageList[agePre[0].argmax()]
               
               label="{},{}".format(gender,age)
               cv.putText(frame,label,(bbox[0],bbox[1]-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
         result.write(frame)
         cv.imshow("Age gender",frame)
         k = cv.waitKey(1)
         if k == ord("q"):
            break
      else: break
   # result.release()
   video.release()
   cv.destroyAllWindows()
if __name__ == "__main__":
   main()
