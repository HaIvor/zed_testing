import pyzed.sl as sl
import cv2

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use 720p resolution
    init_params.camera_fps = 30  # Set FPS

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open the ZED camera")
        exit(-1)

    # Create objects to store images
    image_zed = sl.Mat()

    # Capture loop to display the camera feed
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image from the ZED camera
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)

            # Convert ZED Mat to an OpenCV-compatible format (RGB image)
            frame = image_zed.get_data()

            # Display the frame using OpenCV
            cv2.imshow('ZED Camera', frame)

            # Exit loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the camera and OpenCV window
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
