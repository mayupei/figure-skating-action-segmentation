import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from file_paths import VIDEO_PATH, FIGURE_PATH

BONES = [
    (0, 1),  # Nose → Left Eye
    (0, 2),  # Nose → Right Eye
    (1, 3),  # Left Eye → Left Ear
    (2, 4),  # Right Eye → Right Ear
    (5, 6),  # Left Shoulder → Right Shoulder
    (5, 7),  # Left Shoulder → Left Elbow
    (7, 9),  # Left Elbow → Left Wrist
    (6, 8),  # Right Shoulder → Right Elbow
    (8, 10),  # Right Elbow → Right Wrist
    (11, 12),  # Left Hip → Right Hip
    (5, 11),  # Left Shoulder → Left Hip
    (6, 12),  # Right Shoulder → Right Hip
    (11, 13),  # Left Hip → Left Knee
    (13, 15),  # Left Knee → Left Ankle
    (12, 14),  # Right Hip → Right Knee
    (14, 16),  # Right Knee → Right Ankle
]

def set_equal_axes(ax, data):
    max_range = np.ptp(data.reshape(-1, 3), axis=0).max() / 2
    center = np.mean(data.reshape(-1, 3), axis=0)
    for axis, mid in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        axis(mid - max_range, mid + max_range)
        
        
def create_video_with_predictions(skeletons, preds, labels, file_name):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Skeleton Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Prepare a placeholder for joints and bones
    scatter = ax.scatter([], [], [], c="red")
    lines = [ax.plot([], [], [], "b")[0] for _ in BONES]
    text = ax.text(0.05, 0.9, 0.9, "", transform=ax.transAxes, fontsize=12)
    
    # set equal axes
    set_equal_axes(ax, skeletons)
    
    # Update function
    def update(frame_idx):
        joints = skeletons[frame_idx]
        scatter._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
        for i, (a, b) in enumerate(BONES):
            lines[i].set_data([joints[a, 0], joints[b, 0]], [joints[a, 1], joints[b, 1]])
            lines[i].set_3d_properties([joints[a, 2], joints[b, 2]])
        text.set_text(f"pred: {preds[frame_idx]}; true: {labels[frame_idx]}")
        return [scatter]
    
    anim = FuncAnimation(fig, update, frames=skeletons.shape[0], interval=30, blit=False)
    anim.save(os.path.join(VIDEO_PATH, f"{file_name}.mp4"), writer='ffmpeg', fps=30)
    
    
def generate_skeleton_plot_one_frame(frame, title, file_name):    
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # set equal axes
    set_equal_axes(ax, frame)
    
    ax.scatter(frame[:,0], frame[:,1], frame[:,2], c = "blue")

    for b in BONES:
        ax.plot([frame[b[0], 0], frame[b[1], 0]],
                 [frame[b[0], 1], frame[b[1], 1]],
                  [frame[b[0], 2], frame[b[1], 2]], c = "red")
        
    plt.show()
    fig.savefig(os.path.join(FIGURE_PATH, file_name), bbox_inches="tight")