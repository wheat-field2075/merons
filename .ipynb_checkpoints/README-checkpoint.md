### 2023/08/14: model_testing ###
- some edits to model_testing for AP calculations
- minor bugfixes

### 2023/08/08: fall cleaning
- condensed many files in modules and added a little bit of documentation on how to use
- made an actually presentable model_testing notebook

### 2023/07/28: ???
- tried many different loss methods, ended up sticking with generalized focal loss
- replace vanilla UNet with Hourglass network
- more data?? how much should I annotate. probably should max it out

### 2023/06/09: focal loss
- tried out elastic transforms, grid distortions, optical distortions, Gaussian noise on UNet method, didn't help perf.
- tried using focal loss w/ pre-trained ResNet and VGGNet
- currently using UNet w/ focal loss, needs debugging

### 2023/05/22: dihedral4
- added augmentations for dihedral-4
- loss on val-data is noticably lower, but fails to yield noticable improvements on the entire patch

### 2023/03/17: negative samples and dataset_process
- should be dated as 2023/05/17
- dataset_preprocess.ipynb now generates patches and stores them on disk, rather than creating patches at runtime and storing in memory. should help w/ scalability.
- added negative samples to training, which are defined as patches w/o a patch center in the mask center. patches are randonly sampled from the images and checked.