### AI-Based Automatic Cinematic Framing
### Shang Ni (s5701147)
___README is a Simplified version, refer to the complete Report for further information:)___

#### Abstract                                                 
Cinematic camera control is a cornerstone of visual storytelling in film, animation, and interactive media, yet remains a labor‑intensive task typically handled by expert artists. While recent deep learning methods automate camera placement and movement from video, they depend heavily on large, annotated video corpora and struggle to generalize to novel character interactions. In this work, we propose a novel framework that learns to predict Toric camera parameters directly from two‑person 3D motion data, bypassing the need for preexisting visual datasets. Our model employs a dual‑stream Transformer to encode each character’s motion, fuses these streams via bidirectional cross‑attention to capture inter‑character dynamics, and incorporates explicit spatial vectors to ground geometric relationships. A lightweight fusion network then regresses per‑frame Toric parameters, yielding smooth, compositionally balanced camera trajectories. To enable training and evaluation, we introduce a new dataset of over 3,400 motion–camera sequences spanning diverse interaction scenarios. Experiments demonstrate that our approach significantly outperforms a strong Example‑Driven Camera baseline and ablated variants in trajectory accuracy, framing quality, and temporal coherence.








