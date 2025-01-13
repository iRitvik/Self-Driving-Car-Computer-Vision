import cv2

# img_file = r"C:\Users\ranar\Desktop\Object Detection Self Driving Cars VSCode\Car Image.jpg"

# video =  cv2.VideoCapture(r"C:\Users\ranar\Desktop\Object Detection Self Driving Cars VSCode\Tesla FSD Autopilot Dashcam Compilation.mp4")
video =  cv2.VideoCapture(r"Tesla FSD Autopilot Dashcam Compilation.mp4")
# video =  cv2.VideoCapture(r"Teslas Avoiding Accidents_Trim.mp4")

car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)

    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    print(cars)

    # Building the Rect Box
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 127), 1)

    cv2.imshow("Car and Pedestrian Tracking",  frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
video.release()

