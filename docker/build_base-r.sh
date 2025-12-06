#!/usr/bin/env bash

echo "Build the docker"

# Parameters
user_name="rkrispin"
image_label="baser"
r_major=4
r_minor=4
r_patch=0
quarto_ver="1.8.26"
python_ver="3.10"
ruff_ver="0.14.0"
venv_name="the-forecaster-dev"
requirements="requirements-base.txt"

# Setting the image name
ver=${r_major}.${r_minor}.${r_patch}
tag="0.0.1"
docker_file=Dockerfile.base-r
image_name=$user_name/$venv_name-baser$ver:$tag

echo "Image name: $image_name"

# Build
docker build . \
  -f $docker_file --progress=plain \
  --platform linux/amd64,linux/arm64 \
  --build-arg PYTHON_VER=$python_ver \
  --build-arg R_VERSION_MAJOR=$r_major \
  --build-arg R_VERSION_MINOR=$r_minor \
  --build-arg R_VERSION_PATCH=$r_patch \
  --build-arg QUARTO_VERSION=$quarto_ver \
  --build-arg VENV_NAME=$venv_name \
  --build-arg RUFF_VER=$ruff_ver \
  --build-arg REQUIREMENTS=$requirements \
   -t $image_name

# Push
if [[ $? = 0 ]] ; then
echo "Pushing docker..."
docker push $image_name
else
echo "Docker build failed"
fi