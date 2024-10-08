#!/usr/bin/env bash
docker build -t shubham/omsense .
docker run -it -v ./:/CG_CNN shubham/omsense

# echo "setting up dependency installs, may take up to 20 min"
# conda clean --all -y
# conda env create --prefix ./conda_env -f conda_env.yaml
# echo "starting artifact run"
# cd artifact && conda run -p ../conda_env python CG_CNN_inference.py

