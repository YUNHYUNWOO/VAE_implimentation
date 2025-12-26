# base image는 pytorch
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y sudo

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USER_GID /workspace \
    && chmod -R 775 /workspace

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

SHELL ["/bin/bash", "-lc"]