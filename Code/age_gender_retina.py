from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt


# faces_extract = RetinaFace.extract_faces(img_path = img_path, align = True)
# print("faces: ",len(faces))
# print("faces_extract: ",len(faces_extract))


# print(faces.keys())
def facebox(frame):
   faces = RetinaFace.detect_faces(frame)
   # faces_extract = RetinaFace.extract_faces(img_path = frame, align = True)
   # print("face extract: ",faces_extract)
   bboxs = []
   for key in faces.keys():
      thongtin = faces[key]
      print('Thong tin: ',thongtin)
      facial_area = thongtin["facial_area"]
      confidence = thongtin['score']
      if confidence > 0.9:
         x1 = facial_area[0]
         y1 = facial_area[1]
         x2 = facial_area[2]
         y2 = facial_area[3]
         bboxs.append([x1,y1,x2,y2])
         cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),5)
   return frame,bboxs  

 
def set_saved_video(input_video, output_video, size):
   fourcc = cv2.VideoWriter_fourcc(*"MP4V")
   fps = int(input_video.get(cv2.CAP_PROP_FPS))
   print("FPS: ",fps)
   video = cv2.VideoWriter(output_video,fourcc,10,size)
   return video

#add the age model and age proto
agemodel=r"model_config\age_net.caffemodel"
ageproto = r"model_config\age_deploy.prototxt"

# path the gender model and proto for gender prediction
gendermodel = "model_config\gender_net.caffemodel"
genderproto = "model_config\gender_deploy.prototxt"   
     
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

ageNet = cv2.dnn.readNet(agemodel,ageproto)
genderNet = cv2.dnn.readNet(gendermodel,genderproto)

# #Set backend when run model on gpu device
# ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

video = cv2.VideoCapture("img_video\webcam2.mp4")
if (video.isOpened() == False):
   print("Error reading video file")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Width: ",width)
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("height: ",height)


def main():
   result = set_saved_video(video,"result_webcam2.mp4",(width,height))
   while True:
      ret,frame = video.read()
      if ret == True:
         frame, bboxs = facebox(frame)
         for bbox in bboxs:
            #crop face from frame detection
            face = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            # face = cv2.resize(face,(224,224))
            print("face: ",face)
            blob = cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB = False)
            genderNet.setInput(blob)
            gender_predict = genderNet.forward()
            gender = genderList[gender_predict[0].argmax()]
            
            ageNet.setInput(blob)
            agePre =  ageNet.forward()
            age = ageList[agePre[0].argmax()]
            
            label="{},{}".format(gender,age)
            cv2.putText(frame,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
         result.write(frame)
         cv2.imshow("Age gender",frame)
         k = cv2.waitKey(1)
         if k == ord("q"):
            break
      else: 
         break
   # result.release()
   video.release()
   cv2.destroyAllWindows()
if __name__ == "__main__":
   main()

