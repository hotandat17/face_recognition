import cv2
import os

def set_webcam_resolution(video_capture, width, height):
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def capture_images_for_training(output_folder, num_images, width, height):
    video_capture = cv2.VideoCapture(0)
    set_webcam_resolution(video_capture, width, height)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_count = 0
    
    while image_count < num_images:
        # Read the video frame
        ret, frame = video_capture.read()
        cv2.imshow('Chụp ảnh nè', frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            name = input("Nhập tên cho ảnh: ")
            
            image_filename = os.path.join(output_folder, f'{name}.jpg')
            cv2.imwrite(image_filename, frame)
            image_count += 1
            
            print(f'Image {image_count} đã lưu.')
        
        if key == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

output_folder = 'photos'
num_images = 1
desired_width = 3456
desired_height = 5184

capture_images_for_training(output_folder, num_images, desired_width, desired_height)
