from unittest import result
import cv2 as cv

videopath = 'marvel.mp4'

cap = cv.VideoCapture(0)
width_ = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height_ = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
print("w:",width_)
print("h:",height_)
print("fps:",fps)

def set_saved_video(input_video, output_video, size):
   fourcc = cv.VideoWriter_fourcc(*"MP4V")
   fps = int(input_video.get(cv.CAP_PROP_FPS))
   print("FPS: ",fps)
   video = cv.VideoWriter(output_video,fourcc,fps,size)
   return video


def main():
   result = set_saved_video(cap,"webcam2.mp4",(640,640))
   while True:
      ret, frame = cap.read()
      frame = cv.resize(frame,(640,640))
      result.write(frame)
      cv.imshow("frame",frame)
      if cv.waitKey(1) & 0xFF == ord("q"):
         break
   result.release()
   cap.release()
   cv.destroyAllWindows()
   
if __name__ == "__main__":
   main()