# Mice Behavior Tracking

Instructions for setting up and using pose-estimation networks to extract behavioral measurements from videos acquired during widefield calcium imaging experiments.

## Overview

Behavioral features are extracted from videos using pose-estimation networks. We use [Lightning Pose](https://lightning-pose.readthedocs.io/en/latest/source/installation_guide.html) for training and inference, although [DeepLabCut (DLC)](https://github.com/DeepLabCut/DeepLabCut) can also be used.

The general workflow is:

1. Install Lightning Pose.
2. Create a Lightning Pose project.
3. Import existing labelled datasets where available.
4. Add labelled frames from your experimental dataset.
5. Train the pose-estimation network.
6. Run the trained network on experimental videos.
7. Extract behavioral measurements from the predicted coordinates.

> **Note:** Lightning Pose requires a computer or server with a compatible GPU for training.

---

## Lightning Pose

We use [Lightning Pose](https://lightning-pose.readthedocs.io/en/latest/source/installation_guide.html) for pose estimation. If you are more comfortable using DeepLabCut (DLC), the same general workflow can also be implemented using DLC.

When installing Lightning Pose, use:

**Option A: Standard Installation (Standard)**

For general information on training networks, refer to the [Lightning Pose documentation](https://lightning-pose.readthedocs.io/en/latest/source/lightning_pose_3d.html).

The sections below describe the additional steps and datasets used for our specific tracking networks.

---

# Pupil Tracking Network

The pupil tracking network was initialized using a pre-trained DLC pupil model originally developed by the [Schroeder Lab](https://github.com/Schroeder-Lab/EyeVideoAnalysis).

The original model and labelled dataset are available from [Figshare](https://sussex.figshare.com/articles/software/Model_to_detect_the_pupil_of_a_mouse_in_videos/24072354/1?file=42232794).

The Schroeder Lab dataset contains approximately 1,000 labelled images from approximately 30 videos. Camera position and lighting varied across videos, making the dataset useful for training a model that can generalize across different experimental setups.

### Training workflow

#### 1. Create a Lightning Pose project

Start a new project in Lightning Pose.

#### 2. Import the Schroeder Lab labelled dataset

Download the [labelled dataset](https://sussex.figshare.com/articles/software/Model_to_detect_the_pupil_of_a_mouse_in_videos/24072354/1?file=42232794). The DLC-labelled dataset can be converted to the Lightning Pose format using the `dlc2lp.py` conversion script.

First, navigate to the Lightning Pose installation directory:

```bash
cd <path/to/lightning-pose>
```

Then run:

```bash
python scripts/converters/dlc2lp.py \
    --dlc_dir=<path/to/MousePupil-SchroederLab-2023-08-02_upload> \
    --lp_dir=<path/to/LPProjects/pupilmodel>
```

This will copy the labelled images and annotations into the Lightning Pose project.

#### 3. Add labelled images from your experimental dataset

Manually label frames from your own videos and add them to the Lightning Pose project.



As a starting point, include:

* **At least ~100 labelled images from your experimental dataset**, or
* **~10–25 images per video** when multiple videos are available.

> **Important:** Include examples of difficult frames, such as highly dilated or highly constricted pupils, rather than selecting only frames where the pupil is easy to identify.

Lightning Pose can automatically select representative frames from videos based on variation within the video. However, if it doesn't include frames of varying pupil size, then you can also manually add frames that are difficult.



#### 4. Train the network

Train a new Lightning Pose model using the combined dataset:

* Schroeder Lab labelled images
* Your experimentally labelled images

The resulting model should retain the generalization provided by the larger Schroeder Lab dataset while being adapted to the specific appearance and recording conditions of your experimental setup.

---

# Body Tracking Network

Unlike pupil tracking, we currently do not use a pre-existing labelled dataset for body tracking because available datasets do not sufficiently match our experimental setup.

The body tracking model is therefore trained using manually labelled frames from our experimental videos.

### Training workflow

#### 1. Create a Lightning Pose project

Start a new project in Lightning Pose.

#### 2. Label frames from your experimental dataset

Manually label frames from your experimental videos.

As a starting point, use approximately:

**50–100 labelled images per mouse.**

Try to sample frames that cover the range of positions and postures encountered during the experiment.

Include variation in:

* Mouse position
* Body orientation
* Posture
* Movement
* Lighting
* Occlusions or partially obscured body parts

#### 3. Train the network

Train a new Lightning Pose model using the manually labelled dataset.

---

# Behavioral Measurements

Once a pose-estimation network has been trained, it can be used to predict the coordinates of each tracked keypoint throughout the experimental videos.

These coordinates can then be used to calculate behavioral measurements such as:

* Pupil size
* Body position
* Movement
* Framewise displacement
* Other behavioral features derived from keypoint coordinates

For instructions on extracting behavioral measurements from the predicted coordinates, refer to the `README.md` file in the [`code`](https://github.com/JoshuaBTan/mice_behaviour/tree/main/code) folder.
