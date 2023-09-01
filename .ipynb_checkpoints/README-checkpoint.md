### message to Steven
Hey there Steven, cleaned up the code a bit and this should be able to run. Input data and annotations should be put into ./dataset; however, once you've completed annotations for anti-merons, it'd be wise to put that into a new folder. To train a model, check out model_testing.ipynb. To run longer training sessions, convert model_testing.ipynb to a Python file and submitting a slurm job.

For next steps, consider adding dropout layers in training. In addition, it'd be nice to develop an anti-meron detector and developing a tool to visualize when merons turn into anti-merons (and vice versa). For annotations, try using the track feature in cvat.ai. Professor Wei recommended https://streamlit.io/ to package the final result nicely.

Lemme know if you'd need help or some explanations/advice, and happy coding!

### 2023/08/08: fall cleaning 2
- split up the datatools file into numerous datatool* files to make them more navigatable
    - datatools0: basic functions to open JPEG images as numpy files
    - datatools1: functions to create a temporary dataset from a root directory with images and annotations
    - datatools2: a function to apply transformations to a dataset and a custom Dataset class
    - datatools3: functions to calculate precision and recall on an image; treats image as a collection of input patches and 
        stitches the resulting prediction patches together

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