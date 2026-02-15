import cv2
import os

IPM_VIDEO_DIRECTORY='./dashboard-IPM-only.mp4'


def extract_frames(input_path,output_folder_path):
    frame_count = 0
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Could not open from path {input_path}")

    while True:
        ret, frame = cap.read()


        if not ret:
            break
        frame_filename = os.path.join(output_folder_path,f"frame-{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Extraction complete {frame_count:04d} total frames extracted.")




if __name__=='__main__':
    extract_frames(IPM_VIDEO_DIRECTORY,'./extracted-frames')