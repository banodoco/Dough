import os
import subprocess
import cv2
import moviepy.editor as mp
import ffmpeg


def get_video_info(file_path):
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate,duration,bit_rate -of default=noprint_wrappers=1:nokey=1 {file_path}"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8").split("\n")
    info = {
        "codec": output[0],
        "width": int(output[1]),
        "height": int(output[2]),
        "frame_rate": output[3],
        "duration": float(output[4]),
        "bit_rate": int(output[5]),
    }
    return info


def process_video(input_video, desired_speed_change, output_video):
    # Read input video
    clip = mp.VideoFileClip(input_video)
    input_video_duration = clip.duration

    # Apply speed change
    output_clip = clip.fx(mp.vfx.speedx, desired_speed_change)

    # Save output video
    output_clip.write_videofile(output_video, codec="libx264", preset="fast")

    # Close clips
    clip.close()
    output_clip.close()


if __name__ == "__main__":
    input_video = "videos/how_would_you_tell_2/assets/videos/0_raw/s702998f1ux8f107.mp4"
    desired_speed_change = 0.9090909090909091
    output_video = "videos/how_would_you_tell_2/assets/videos/1_final/720dj1i64cc04ytj.mp4"

    process_video(input_video, desired_speed_change, output_video)

    input_video_info = get_video_info(input_video)
    output_video_info = get_video_info(output_video)

    print("Input video information:")
    for key, value in input_video_info.items():
        print(f"{key}: {value}")

    print("\nOutput video information:")
    for key, value in output_video_info.items():
        print(f"{key}: {value}")