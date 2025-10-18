import pickle
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from matplotlib.animation import FuncAnimation
import os

# Load the pickle file
with open('arabic/pose_data_isharah1000_hands_lips_body_May12.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# Target IDs to process
target_ids = ['14_0004', '14_0001']

# Constants
NUM_LIPS = 19
FRAME_SKIP = 8
NORMALIZE = True  # Set to False to see original behavior
SCALE_NORMALIZE = True  # Also normalize scale

def normalize_frame(frame, num_lips=19):
    """Normalize a frame to center it and optionally scale it."""
    # Body keypoints start after hands and lips
    body_start = 42 + num_lips
    body = frame[body_start:]
    
    body_center = np.mean(body, axis=0)
    
    # Center the frame
    normalized = frame - body_center
    
    if SCALE_NORMALIZE:
        distances = np.linalg.norm(body - body_center, axis=1)
        max_distance = np.max(distances)
        
        if max_distance > 0:
            scale_factor = 200 / max_distance
            normalized = normalized * scale_factor
    
    return normalized

def draw_connections(part, connections, offset, color, combined):
    for conn in connections:
        start_idx, end_idx = conn
        start = offset + start_idx
        end = offset + end_idx
        if start < len(combined) and end < len(combined):
            x1, y1 = combined[start]
            x2, y2 = combined[end]
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=2)

def draw_points(part, offset, color):
    for i, (x, y) in enumerate(part):
        plt.scatter(x, y, color=color, s=30)

def animate_frame(frame_idx, keypoints, fig, ax):
    ax.clear()
    
    frame = keypoints[frame_idx]
    
    if NORMALIZE:
        frame = normalize_frame(frame, NUM_LIPS)
    
    rh = frame[0:21]
    lh = frame[21:42]
    lips = frame[42:42+NUM_LIPS]
    body = frame[42+NUM_LIPS:]
    
    combined = np.concatenate([rh, lh, lips, body], axis=0)
    
    rh_offset = 0
    lh_offset = 21
    lips_offset = 42
    body_offset = 42 + NUM_LIPS
    
    # Draw each component
    draw_connections(rh, mp.solutions.hands.HAND_CONNECTIONS, rh_offset, 'red', combined)
    draw_connections(lh, mp.solutions.hands.HAND_CONNECTIONS, lh_offset, 'blue', combined)
    draw_connections(body, mp.solutions.holistic.POSE_CONNECTIONS, body_offset, 'purple', combined)
    
    draw_points(rh, rh_offset, 'red')
    draw_points(lh, lh_offset, 'blue')
    draw_points(lips, lips_offset, 'green')
    draw_points(body, body_offset, 'purple')
    
    # Set fixed axis limits for stable view
    if NORMALIZE:
        ax.set_xlim(-300, 300)
        ax.set_ylim(-300, 300)
    
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

# Create output directory
output_dir = 'pose_animations'
os.makedirs(output_dir, exist_ok=True)

# Process each target ID
for sample_id in target_ids:
    if sample_id in data_dict:
        print(f"Processing {sample_id}...")
        
        keypoints = data_dict[sample_id]['keypoints']
        keypoints = keypoints[::FRAME_SKIP]
        num_frames = keypoints.shape[0]
        
        print(f"Using {num_frames} frames (normalized: {NORMALIZE})")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        anim = FuncAnimation(
            fig, 
            animate_frame, 
            frames=num_frames,
            fargs=(keypoints, fig, ax),
            interval=100,
            repeat=True
        )
        
        gif_path = os.path.join(output_dir, f'{sample_id}_pose_animation.gif')
        print(f"Saving GIF to {gif_path}...")
        anim.save(gif_path, writer='pillow', fps=10)
        
        plt.close(fig)
        print(f"Saved {sample_id}")
        
    else:
        print(f"ID {sample_id} not found in data")

print("All animations saved!")