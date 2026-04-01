## Knowledge Distillation (CIFAR-10)

This repository demonstrates a simple and reproducible knowledge distillation pipeline:

Teacher: ResNet-50  
Student: ResNet-18  
Dataset: CIFAR-10

This is a small-scale experiment designed for quick testing and learning purposes.  
The training requires **less than 4GB GPU memory**, making it accessible for most users with a GPU.

Batch size can be adjusted depending on your GPU memory.

Before running the experiment, please make sure your Python and PyTorch environments are properly installed.
## Requirements

- Python 3.9+
- PyTorch
- torchvision
- tqdm
## Environment Setup (Linux / WSL)

### 1. Update system packages

# Update package list
sudo apt update

# Install basic dependencies
sudo apt install -y python3 python3-pip python3-venv git


### 2. Create Python virtual environment

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate


### 3. Upgrade pip

# Upgrade pip to latest version
pip install --upgrade pip


### 4. Install PyTorch

# Install PyTorch (GPU version recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# If GPU is not available, install CPU version
# pip install torch torchvision


### 5. Install additional dependencies

pip install tqdm


### 6. Verify installation

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
