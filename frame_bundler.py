import cv2
import os

def extract_frame_sets(video_path, output_dir, target_fps=1, frames_per_set=3):
    """
    Extracts frames from a video and groups them into numbered sets.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get the original framerate of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate how many frames to skip to match our target_fps
    frame_interval = int(round(original_fps / target_fps))

    frame_count = 0
    extracted_count = 0
    set_count = 1

    print(f"Extracting {target_fps} FPS into sets of {frames_per_set}...")

    while True:
        success, frame = cap.read()
        if not success:
            break # End of video

        # If this frame lands on our target interval, save it
        if frame_count % frame_interval == 0:
            
            # Optional but recommended: Resize to 480p to save memory/processing time
            frame = cv2.resize(frame, (854, 480))

            # Determine where this frame belongs in the current 3-image set
            frame_in_set = (extracted_count % frames_per_set) + 1

            # Format: set001_frame1.jpg
            filename = os.path.join(output_dir, f"set{set_count:03d}_frame{frame_in_set}.jpg")
            cv2.imwrite(filename, frame)

            extracted_count += 1

            # Once we hit our limit (e.g., 3 frames), tick the set counter up
            if extracted_count % frames_per_set == 0:
                set_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Extracted {extracted_count} total frames across {set_count - 1} complete sets.")

# --- How to run it ---
# Change "test_clip.mp4" to your actual video file path
if __name__ == "__main__":
    extract_frame_sets(
        video_path="test_clip.mp4", 
        output_dir="gemma_test_frames", 
        target_fps=1,        # 1 frame per second
        frames_per_set=3     # Groups of 3
    )