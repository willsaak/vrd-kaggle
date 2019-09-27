import cv2

path = "/mnt/renumics-research/datasets/vis-rel-data/test_img/test/39dd3c9d159c3094.jpg"
img = cv2.imread(path)
cv2.imshow('sadf', img)
cv2.waitKey(0)