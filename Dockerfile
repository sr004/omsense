FROM continuumio/anaconda3:2024.06-1

COPY conda_env.yaml /

VOLUME /CG_CNN
WORKDIR /CG_CNN/artifact
RUN conda env create -p /conda_env -f /conda_env.yaml
CMD ["conda", "run", "-p", "/conda_env", "python", "CG_CNN_inference.py"]
