from statistics import mode
from retinaface import RetinaFace
import cv2 as cv
import matplotlib.pyplot as plt
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
model = get_model("resnet50_2020-07-20",max_size=2048)
model.eval()

img = cv.imread("GD.jpg")
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

annotation = model.predict_jsons(img)
print("annotation: ",annotation)
cv.imshow("retina pytorch",annotation)
cv.waitKey(0)
cv.destroyAllWindows()
# img_path = "GD.jpg"
# img = cv.imread(img_path)
# # face_ex = RetinaFace.extract_faces(img_path = img_path,align = True)
# # print("face ex: ",face_ex)
# # for key in face_ex:
# #    cv.imshow("face_ex",key[:,:,::-1])
# #    cv.waitKey(0)
# # cv.destroyAllWindows()

# faces = RetinaFace.detect_faces(img_path)
# #    # faces_extract = RetinaFace.extract_faces(img_path = frame, align = True)
# #    # print("face extract: ",faces_extract)
# bboxs = []
# for key in faces.keys():
#    thongtin = faces[key]
#    print('Thong tin: ',thongtin)
#    facial_area = thongtin["facial_area"]
#    confidence = thongtin['score']
#    if confidence > 0.95:
#       x1 = facial_area[0]
#       y1 = facial_area[1]
#       x2 = facial_area[2]
#       y2 = facial_area[3]
#       bboxs.append([x1,y1,x2,y2])
      
# print("bboxs: ",bboxs)

# for bbox in bboxs:
#    face = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
#    face_align = RetinaFace.extract_faces(img_path = face, align = True)
#    print("face crop:",face)
#    print("face_align: ",face_align)
#    cv.imshow("face crop",face_align)
#    cv.waitKey(0)
   
# cv.destroyAllWindows()
