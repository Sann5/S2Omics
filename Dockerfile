# Use the same base image
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Copy the requirements file (equivalent to %files)
COPY requirements.txt .

# Since pytorch and python are installed in the base env (from the image)
# and we want to work with that, we need to use --break-system-packages
# to install into the base env
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt