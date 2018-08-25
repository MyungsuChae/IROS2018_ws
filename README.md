The code of End-to-end multimodal emotion and gender recognition with dynamic weights of joint loss. 
## About
Submitted to arXiv (cs.LG, stat.ML).
Submitted to IROS 2018 Workshop on Crossmodal Learning for Intelligent Robotics.

### Authors
Myungsu Chae, Tae-Ho Kim, Young Hoon Shin, Jun-Woo Kim, Soo-Young Lee (advisor) in KAIST Institute for Artificial Intelligence.

### Contribution
The research shows the strength (high generalizability / low negative log-likelihood) of dynamic weights for minimizing joint loss at first in emotion and gender recognition task.


## Getting started

### Install prerequisites

```
pip install -r requirements.txt
```

### Download dataset
Interactive Emotional Dyadic Motion Capture database (IEMOCAP) from SAIL Lab in USC. [https://sail.usc.edu/iemocap/]


### Preprocess
```
python process.py #WARNING: It uses CPU very much.
```

### Train 
```
CUDA_VISIBLE_DEVIES=0 python train.py [--modal MODAL] [--loss LOSS] [--stl_dim STL_DIM]
```
* `MODAL` is one of `audio`, `video`, or `multi`.
* `LOSS` is one of `Static` or `Joint`.
* In case `LOSS` is `Static`, you should specify `STL_DIM` which controls the weights between tasks. It sets to `-1` for balancing uniformly by default. `0` and `1` are emotion classification and gender classification, respectively.
* By default, model saved to `./result

