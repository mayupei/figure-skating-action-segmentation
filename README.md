# Action Segmentation for Figure Skating Competition Videos: A Skeleton-based Approach

## Introduction
Figure skating is a sport in which skaters execute pre-planned technical elements (e.g., jumps, spins) within a choreographed routine set to music. In figure skating judging, judges often need to replay a technical element to determine its difficulty and execution. Under the current system, a replay operator is responsible for marking the start and end time of each element as it's being performed, allowing for quick access to replays during the review process. In this project, I aim to automate this process via action segmentation using a deep learning approach.

In a complete pipeline, we need to first extract the 3D skeleton joints from each video frame, capturing the athlete's posture while ignoring irrelevant information such as the background colour and the audience's movement. Then the model assigns each frame a label (e.g., a jump or a spin) and outputs the start and end timestamps for each segment/element. There are already well-established methods available for skeleton extraction. As a result, this project uses the existing skeleton-based figure skating video datasets and focuses on building the action segmentation model.

## Data and Summary Statistics
The datasets used in this project are [MCFS](https://shenglanliu.github.io/mcfs-dataset/) and [MMFS](https://github.com/dingyn-Reno/MMFS/tree/main). These two datasets were developed by the same team and both used videos from the 2017-2019 World Figure Skating Championships.

MCFS contains 271 videos of single-skater competition routines. Each video is 162–285 s long (≈2.7–4.75 min) and recorded at 30 fps. They extracted 3D pose skeletons with OpenPose (BODY_25). The dataset provides per-frame (frame-wise) annotations, which are essential for supervised training for action segmentation. However, the pose quality is imperfect: X% of frames have at least one missing joint, and Y% of frames have at least X missing joints.

MMFS consists of 1176 videos of single-skater competition routines, among which 222 routines share the same video source as MCFS. Like MCFS, each video is recorded at 30 fps. 3D Skeletons are extracted in the COCO 17-keypoint format. It has a better data quality than NCFS: only X\% of frames have missing joints, as opposed to Y% in MCFS. However, MMFS only provides elements performed in each routine and does not contain per-frame annotations.

In order to use the best data source available, I use the 222 shared routines between MCFS and MMFS. More specifically, I use the skeleton data from MMFS as features and the per-frame annotations from MCFS as labels.
