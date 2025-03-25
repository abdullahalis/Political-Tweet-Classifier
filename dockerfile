# Use an official Conda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment.yml to the container
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Make sure the environment is activated when the container starts
SHELL ["conda", "run", "-n", "cs-418", "/bin/bash", "-c"]

# Set the default command to run an interactive shell
CMD ["bash"]
