# omsense
This repository provides the implementation code for “Don’t Crosstalk to Me: Origami Structure-Augmented Sensing for Scalable Surface Pressure Monitoring” Sensys 24

## Overall results:
To get the end-to-end results clone this repository and run the "./setup_docker.sh" script. The script uses a docker environment to run the inference code using the pre-trained models provided in "../artifact" folder to run the inference on real-world data provided in "../figure_7x11/realworld_data2"  and save the output results in "../output" as "overall_results.png"

## 3D model
"../3D model" folder contains the STL files required for printing the physical surface augmentation structure. 

## artifact
The main source code for training and inference can be found in the "../artifact folder". 

Run "CG_CNN_training.py" file to train a new model from scratch, this script uses the simulation data provided "../figure_7x11". 


Inference code is provided in "CG_CNN_inference.py", the docker command above uses this script to generate the overall results.


We also provide the pre-trained models for OMSense "model_5.pth" and velostat baseline "velostat_model_5.pth".



