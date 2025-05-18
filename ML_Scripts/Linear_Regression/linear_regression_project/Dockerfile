# Use the official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda update -n base -c defaults conda

# Create the environment and install dependencies
COPY requirements.txt /app/requirements.txt
RUN /opt/conda/bin/conda create -n ml_pipeline python=3.9 && \
    /opt/conda/bin/conda run -n ml_pipeline pip install --no-deps --force-reinstall -r /app/requirements.txt && \
    /opt/conda/bin/conda run -n ml_pipeline pip install python-Levenshtein flaml[automl] && \
    /opt/conda/bin/conda clean -afy

# Activate the environment and ensure it's activated in subsequent RUN commands
RUN echo "source /opt/conda/bin/activate ml_pipeline" >> ~/.bashrc

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME=World

# Run main.py when the container launches
CMD ["bash", "-c", "source ~/.bashrc && conda activate ml_pipeline && python src/main.py"]