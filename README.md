# VAE_implimentation

This repository is a structured pipeline for training a **Variational Autoencoder (VAE)** .

**FID (Fréchet Inception Distance)** evaluation and **WandB** logging is provided.

I conducted simple experiments by varying the **KLD weight ($\beta$ of $\beta$-VAE)**, observed that reconstruction quality and sample diversity decreased as the value of $\beta$ was increased. 

All results and logs are archived in our WandB project:

You can see the results in https://wandb.ai/hyunwoo629-hanyang-university/VAEs.

<img width="1548" height="217" alt="image" src="https://github.com/user-attachments/assets/05f578ca-d8a0-4f78-90fe-8b0ef865acc5" />
  
---

## 1. Project Architecture

* **`run.py`**: The central entry point that orchestrates configuration loading, seed initialization, and the execution of the training loop.
* **`model.py`**: Defines the model architecture.
* **`trainer.py`**: Manages the training process and periodic validation.
* **`data.py`**: Handles automated data downloading, preprocessing.
* **`log.py`**: Defines log funtions during Training models.
* **`FID_calculator.py`**: Computes the Fréchet Inception Distance to quantitatively measure the quality of generated images.

---

## 2. Installation & Setup

You can set up the environment using the provided Docker configuration. 

**NVIDIA Container Toolkit** must be installed on your host machine to utilize the GPU.

### Option 1: Using Docker CLI
Build the image from the `.devcontainer` directory and run it with full GPU access.

```bash
# 1. Build the Docker image
docker build -t env:latest .devcontainer/

# 2. Run the container
# The --gpus all flag is mandatory for GPU acceleration
docker run --gpus all -it --rm \
    -v ./:/workspace/VAE \
    env:latest
```

### Option 2: Using VS Code Dev Containers
1. Open the project folder in **VS Code**.
2. Ensure the **Dev Containers** extension is installed.
3.  Press `Ctrl` + `Shift` + `P` and select **"Reopen in Container"**.
4. The environment will be automatically built and configured based on the `.devcontainer` settings.


Once the setup is complete, **make sure to update the .env file with your WandB project details.**

---

## 3. How to Run

To start a training session, use the `-n` flag followed by the name of your configuration file (located in the `config/` directory, without the `.yaml` extension).

```bash
python -m run -n [YOUR_CONFIG_NAME]
