import random
import cv2
import numpy as np
from ultralytics import YOLO
import pyzed.sl as sl

def main():
    # Load YOLOv8 nano model
    model = YOLO("weights/yolov8n.pt").to("cuda")
    print(f"Device being used in model: {model.device}")

    # Load class names (COCO dataset)
    with open("utils/coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    # Generate random colors for each class
    detection_colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in class_list
    ]

    # Initialize the ZED camera
    zed = sl.Camera()

    # Set initialization parameters for ZED camera
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 720p resolution
    init_params.camera_fps = 30  # 30 FPS

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Cannot open ZED camera")
        exit()

    # Create an object to store images from the ZED camera
    image_zed = sl.Mat()

    # Main loop for real-time detection
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image from the ZED camera
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)

            # Convert ZED Mat to an OpenCV-compatible format (RGBA to RGB)
            frame = image_zed.get_data()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from RGBA to RGB

            # Predict using YOLOv8 model
            detect_params = model.predict(source=frame_rgb, conf=0.45, save=False)

            # Check if any detections are made
            if len(detect_params[0].boxes) != 0:
                for box in detect_params[0].boxes:
                    # Move tensors to CPU before converting to numpy
                    clsID = int(box.cls.cpu().numpy()[0])  # Class ID
                    conf = box.conf.cpu().numpy()[0]  # Confidence score
                    bb = box.xyxy.cpu().numpy()[0]  # Bounding box coordinates

                    # Draw the detection box
                    cv2.rectangle(
                        frame_rgb,
                        (int(bb[0]), int(bb[1])),
                        (int(bb[2]), int(bb[3])),
                        detection_colors[clsID],
                        3,
                    )

                    # Display class name and confidence score
                    label = f"{class_list[clsID]} {conf:.2f}"
                    cv2.putText(
                        frame_rgb,
                        label,
                        (int(bb[0]), int(bb[1]) - 10),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

            # Display the frame in real-time
            cv2.imshow("ZED Camera YOLOv8 Detection", frame_rgb)

            # Press 'q' to quit the real-time detection
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
