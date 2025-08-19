# Action Segmentation for Figure Skating Competition Videos: A Skeleton-based Approach

## Introduction
Figure skating is a sport in which skaters execute pre-planned technical elements (e.g., jumps, spins) within a choreographed routine set to music. In figure skating judging, judges often need to replay a technical element to determine its difficulty and execution. Under the current system, a replay operator is responsible for marking the start and end time of each element as it's being performed, allowing for quick access to replays during the review process. In this project, I aim to automate this process via action segmentation using a deep learning approach.

In a complete pipeline, 3D skeleton joints need to first be extracted from each video frame, capturing the athlete's posture while ignoring irrelevant information such as the background colour and the audience's movement. Then the model assigns each frame a label (e.g., a jump or a spin) and outputs the start and end timestamps for each segment/element. There are already well-established methods available for skeleton extraction. As a result, this projects uses the existing skeleton-based figure skating video datasets and focuses on building the action segmentation model.

## Data
The datasets used in this project are [MCFS](https://shenglanliu.github.io/mcfs-dataset/) and [MMFS](https://github.com/dingyn-Reno/MMFS/tree/main).
