import os
import tempfile
from moviepy.editor import VideoFileClip, vfx

class VideoProcessor:
    @staticmethod
    def update_video_speed(video_location, desired_duration):
        clip = VideoFileClip(video_location)

        return VideoProcessor.update_clip_speed(clip, desired_duration)
    
    @staticmethod
    def update_video_bytes_speed(video_bytes, desired_duration):
        # Use a context manager for the temporary file to ensure it's deleted when done
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb') as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name  # Store the file name to delete later

        # Process the video file
        with VideoFileClip(temp_file_path) as clip:
            result = VideoProcessor.update_clip_speed(clip, desired_duration)

        # After processing and closing the clip, it's safe to delete the source temp file
        os.remove(temp_file_path)

        return result

    @staticmethod
    def update_clip_speed(clip: VideoFileClip, desired_duration):
        # Use a context manager to ensure temporary file for output is deleted when done
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb') as temp_output_file:
            temp_output_path = temp_output_file.name  # Store the file name for later use

        # if animation_style == AnimationStyleType.DIRECT_MORPHING.value:
        #     clip = clip.set_fps(120)

        #     # Calculate the number of frames to keep
        #     input_duration = clip.duration
        #     total_frames = len(list(clip.iter_frames()))
        #     target_frames = int(total_frames * (desired_duration / input_duration))

        #     # Determine which frames to keep
        #     keep_every_n_frames = total_frames / target_frames
        #     frames_to_keep = [int(i * keep_every_n_frames)
        #                     for i in range(target_frames)]

        #     # Create a new video clip with the selected frames
        #     output_clip = concatenate_videoclips(
        #         [clip.subclip(i/clip.fps, (i+1)/clip.fps) for i in frames_to_keep])

        #     output_clip.write_videofile(filename=temp_output_file.name, codec="libx265")


        # Apply desired video speed change and write to the temporary output file
        input_video_duration = clip.duration
        desired_speed_change = float(input_video_duration) / float(desired_duration)
        print("Desired Speed Change: " + str(desired_speed_change))
        output_clip = clip.fx(vfx.speedx, desired_speed_change)
        output_clip.write_videofile(filename=temp_output_path, codec="libx264", preset="fast")

        # Read the processed video bytes from the temporary output file
        with open(temp_output_path, 'rb') as f:
            video_bytes = f.read()

        # Now it's safe to delete the output temp file since its content is already read
        os.remove(temp_output_path)

        return video_bytes
