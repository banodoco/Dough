import os
import tempfile
from moviepy.editor import concatenate_videoclips, TextClip, VideoFileClip, vfx

from shared.constants import AnimationStyleType

class VideoProcessor:
    @staticmethod
    def update_video_speed(video_location, animation_style, desired_duration):
        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb')
        if animation_style == AnimationStyleType.DIRECT_MORPHING.value:
            # Load the video clip
            clip = VideoFileClip(video_location)

            clip = clip.set_fps(120)

            # Calculate the number of frames to keep
            input_duration = clip.duration
            total_frames = len(list(clip.iter_frames()))
            target_frames = int(total_frames * (desired_duration / input_duration))

            # Determine which frames to keep
            keep_every_n_frames = total_frames / target_frames
            frames_to_keep = [int(i * keep_every_n_frames)
                            for i in range(target_frames)]

            # Create a new video clip with the selected frames
            output_clip = concatenate_videoclips(
                [clip.subclip(i/clip.fps, (i+1)/clip.fps) for i in frames_to_keep])

            output_clip.write_videofile(filename=temp_output_file.name, codec="libx265")

        # modifying speed for any other animation method
        else:
            clip = VideoFileClip(video_location)
            input_video_duration = clip.duration
            desired_speed_change = float(
                input_video_duration) / float(desired_duration)

            print("Desired Speed Change: " + str(desired_speed_change))

            # Apply the speed change using moviepy
            output_clip = clip.fx(vfx.speedx, desired_speed_change)
            
            output_clip.write_videofile(filename=temp_output_file.name, codec="libx264", preset="fast")
        
        with open(temp_output_file.name, 'rb') as f:
            video_bytes = f.read()

        if temp_output_file:
            os.remove(temp_output_file.name)

        return video_bytes