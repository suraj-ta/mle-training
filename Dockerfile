# Use Miniforge as the base image
FROM condaforge/miniforge3

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . .

# Ensure Conda commands work by explicitly using bash
SHELL ["/bin/bash", "-c"]

# Create the Conda environment from environment.yaml
RUN conda env create -f conda/environment.yml

# Activate the Conda environment and ensure it is used by default
ENV PATH /opt/conda/envs/mle-dev/bin:$PATH
ENV CONDA_DEFAULT_ENV=mle-dev

# Set PYTHONPATH so Python finds `src/` correctly
ENV PYTHONPATH="/app/src"

# Run the application inside the Conda environment
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 & sleep 5 && conda run -n mle-dev python src/nonstandardcode/main.py"]
