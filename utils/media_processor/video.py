import cv2, os

def resize_video(input_path, output_path, width, height, crop_type=None, output_format='mp4'):
    # Read input video
    input_video = cv2.VideoCapture(input_path)
    if not input_video.isOpened():
        raise ValueError(f"Could not open the video file: {input_path}")

    # Get source video properties
    src_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate aspect ratios
    src_aspect_ratio = src_width / src_height
    target_aspect_ratio = width / height

    if target_aspect_ratio > src_aspect_ratio:
        # Scale to target width, maintaining aspect ratio
        new_width = width
        new_height = int(src_height * (width / src_width))
    else:
        # Scale to target height, maintaining aspect ratio
        new_width = int(src_width * (height / src_height))
        new_height = height

    # Determine the crop type based on the input dimensions, if not provided
    if crop_type is None:
        width_diff = abs(src_width - width) / src_width
        height_diff = abs(src_height - height) / src_height
        crop_type = 'top_bottom' if height_diff > width_diff else 'left_right'

    # Calculate crop dimensions
    if crop_type == 'top_bottom':
        crop_top = (new_height - height) // 2
        crop_bottom = new_height - crop_top
        crop_left = 0
        crop_right = new_width
    elif crop_type == 'left_right':
        crop_top = 0
        crop_bottom = new_height
        crop_left = (new_width - width) // 2
        crop_right = new_width - crop_left
    else:
        raise ValueError("Invalid crop_type. Must be 'top_bottom' or 'left_right'.")

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # delete the video at the input_path
    os.remove(input_path)

    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(num_frames):
        ret, frame = input_video.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Crop frame
        frame = frame[crop_top:crop_bottom, crop_left:crop_right]

        # Write frame to output video
        output_video.write(frame)

    # Release resources
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()
