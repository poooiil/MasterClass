### AI-Based Automatic Cinematic Framing
### Shang Ni (s5701147)
___README is a Simplified version, refer to the complete Report for further information:)___

#### Abstract                                                 
Cinematic camera control is a cornerstone of visual storytelling in film, animation, and interactive media, yet remains a labor‑intensive task typically handled by expert artists. While recent deep learning methods automate camera placement and movement from video, they depend heavily on large, annotated video corpora and struggle to generalize to novel character interactions. In this work, we propose a novel framework that learns to predict Toric camera parameters directly from two‑person 3D motion data, bypassing the need for preexisting visual datasets. Our model employs a dual‑stream Transformer to encode each character’s motion, fuses these streams via bidirectional cross‑attention to capture inter‑character dynamics, and incorporates explicit spatial vectors to ground geometric relationships. A lightweight fusion network then regresses per‑frame Toric parameters, yielding smooth, compositionally balanced camera trajectories. To enable training and evaluation, we introduce a new dataset of over 3,400 motion–camera sequences spanning diverse interaction scenarios. Experiments demonstrate that our approach significantly outperforms a strong Example‑Driven Camera baseline and ablated variants in trajectory accuracy, framing quality, and temporal coherence.

#### Methodology Overview 
##### 1. Data Collection 

I compiled a  dataset consisting of video clips depicting diverse two-person interactions from films, television series, and stage performances. Each video are selected based on explicit interaction duration, camera motion, and cinematographic quality. 

##### 2. Motion Tracking and Pose Estimation 

Using Intersection-over-Union (IoU) tracking, I identified and consistently tracked the bounding boxes of two main characters across frames. Subsequently, I leveraged MeTRAbs to estimate 2D keypoints and relative 3D poses, and Absolute 3D positions through perspective geometry optimization. 

##### 3. Motion Representation 

Character poses were represented via SMPL-22 joint models, encoded using 6D rotation representations for increased stability. Each character's global translation was appended as a virtual "23rd joint," forming an augmented 120-frame (120×138 dimension) motion sequence. 

##### 4. Gaussian Filtering 

To ensure motion coherence, Gaussian filters were applied along the temporal dimension of joint trajectories, reducing jitter and enhancing animation fluidity. 

##### 5. Dual-Stream Transformer Model 

I employed a Transformer-based neural architecture.

##### 6. Camera Parameter Prediction 

Leveraging Toric coordinates, the model predicts cinematic camera parameters (opening angle, elevation, and azimuth).

##### 7. Visualization of Predicted Camera Positions and Motions 

Convert the predicted normalized camera parameters back into physical space. Then visualize the two characters' motions together with the predicted camera views, demonstrating the AI-generated cinematography decisions, including camera angles, framing, and focal length. 




