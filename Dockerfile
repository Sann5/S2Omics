# Use the same base image
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Copy the requirements file (equivalent to %files)
COPY requirements.txt .

# The base image ships its own opencv build (package name "opencv"). If we
# then install opencv-python from requirements.txt on top of it, pip doesn't
# know the two share the same cv2/ directory, leaving a hybrid/corrupted
# install (AttributeError: module 'cv2.dnn' has no attribute 'DictValue').
# Remove the base image's opencv first so only one clean cv2 build exists.
RUN pip uninstall -y opencv || true && \
    python -c "import site; print(site.getsitepackages()[0])" | xargs -I{} rm -rf {}/cv2*

# Since pytorch and python are installed in the base env (from the image)
# and we want to work with that, we need to use --break-system-packages
# to install into the base env
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt