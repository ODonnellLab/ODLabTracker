import cv2
import sys
import os

def convert_to_grayscale(input_path, output_path=None):
    """
    Convert an RGB AVI video to 8-bit grayscale.
    
    Args:
        input_path: Path to the input AVI file
        output_path: Path for the output file (optional)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_grayscale.avi"
    
    # Open the video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'GREY')  # 8-bit grayscale
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    if not out.isOpened():
        print("Error: Could not create output video file")
        cap.release()
        return False
    
    # Process frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Write the frame
        out.write(gray_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames", end='\r')
    
    print(f"\nProcessed {frame_count}/{total_frames} frames")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Conversion complete! Output saved to: {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_video.avi> [output_video.avi]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_to_grayscale(input_file, output_file)