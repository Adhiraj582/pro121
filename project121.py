import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)
time.sleep(2)

bg = 0
for i in range(60):
    ret, bg = cap.read()

bg = np.flip(bg, axis=1)

frame = cv2.resize(ret, (640, 480))
image = cv2.resize(bg, (640, 480))

while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upper_b = np.array([104, 153, 70])
    lower_b = np.array([30, 30, 0])
    mask = cv2.inRange(frame, lower_b, upper_b)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    f = frame - res
    f = np.where(f == 0, image, f)
    final_output = cv2.addWeighted(res, 1, res, 1, 0)
    output_file.write(final_output)
    cv2.imshow("Magic", final_output)
    cv2.waitKey(1)


cap.release()
# out.release()
cv2.destroyAllWindows()
