# Repository for the Project on: Harris 3D: A robust extension of the Harris operator for interest point detection on 3D meshes

Students: 
Lucas Elbert 
Bjoern Michele 

Explanations for the code: 
- Base.py:  Includes the basic class which can calculate the Harris response values and selects interest points (for all local neighborhood methods) 
- test_experiments.py: Includes the class Experiment and Experiment move. These two classes provide everything to run the experiments. 
- Experiments-Hyperparameter.ipynb: Jupyter notebook which includes the  calculation, saving and visualization of the values for the Hyperparameter selection.
- Experiments-Transformations.ipynb: Jupyter notebook which includes the calculation, saving and visualization of the values for the transformations experiments.
- Basic shapes.ipynb: Jupyter notebook which includes experiments with the basic forms (pyramid, cube), and the visualization of the intermeidate steps of the interest points selection. 

- ply.py: Basic functions to read and write ply files (code taken from the course TPs)
- bunny.ply : Bunny point cloud (taken from the course TPs)

The bunny and the human mesh files can be downloaded here https://drive.google.com/drive/folders/1a2Fd6Vw1SJIwW_M1iUP6pm88QLg7xybT?usp=sharing.
The files should be kept in the Datasets/tosca structure parallel to the 3D Harris folder (not in it). 


Link to the paper original paper: 
http://personales.dcc.uchile.cl/~isipiran/papers/SB11b.pdf
