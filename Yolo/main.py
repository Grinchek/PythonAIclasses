import cv2
from ultralytics import YOLO

imagepath = "2.jpg"

img = cv2.imread(imagepath)

print("This image can be readed")

model = YOLO("yolov8n.pt")

results= model.predict(source = imagepath, conf=0.25)

result_martix = results[0].plot()

cv2.imshow("Result cnowing of objects", result_martix)

cv2.waitKey(0)

cv2.destroyAllWindows()