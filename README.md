# SDF Learner
This program was written and used during the practical part of the bachelor thesis "Placing extraordinary Vertices at positions of high curvature using Signed Distance Fields and Convolutional Neural Networks".
It is used to learn points of high estimated curvature out of given Signed Distance Fields that represent arbitrary triangle meshes using Convolutional Neural Networks.

## Pre-requisites
To run the application, the following programs and packages are needed (preferably the newest available version):
* Python 3.9
* NumPy
* trimesh
* mesh-to-sdf
* PyTorch
* Matplotlib
* seaborn
* torcheval
* torchmetrics
* pyvista
* PyMesh

## Usage
The root folder of the project must contain the folder ```in``` where Signed Distance Fields as ```.bin``` are placed, 
a ```curvature``` folder where the results of the mesh.py script are placed (list of vertices with their corresponding
computed curvature),  an ```out``` folder where the computed ground truth is placed later on, and a ```pred``` folder 
where statistics and predictions are stored during training. Signed Distance Fields should be named of the form 
```sample000000_subdiv.bin``` where the ```000000``` are replaced by numbers in ascending order. The ```main.py``` can
then be invoked using different parameters:
* ```-co```: compute the ground truth (points of high estimated curvature) of one given SDF file inside the ```in```
folder and write to ```out``` folder
* ```-ca```: compute the ground truth of all files inside the ```in``` folder and write to ```out``` folder
* ```-rc```: compute the ground truth from one generated curvature file inside the ```curvature``` folder and write to
```out``` folder
* ```-ra```: compute the ground truth of all generated curvature files inside the ```curvature``` folder and write to
```out``` folder
* ```-v```: visualize either one SDF file of the ```in``` folder, ground truth file of the ```out``` folder, or prediction file of the ```pred``` folder
* ```-t```: train the neural network on a given architecture (```unet2```, ```unet3```, ```seg```) using the SDFs of the ```samples``` folder and the ground truth of the ```out``` folder, results are placed in the ```pred``` folder
* ```-s```: invokes grid search over all defined hyperparameter combinations defined in code (only works together with parameter ```-t```)

The ```mesh.py``` to generate curvature files can be invoked using different parameters:
* ```-co```: compute points of high curvature for one given mesh inside folder ```in``` and write to ```curvature```
folder
* ```-ca```: compute points of high curvature for all mesh files inside folder ```in``` and write to ```curvature```
folder

