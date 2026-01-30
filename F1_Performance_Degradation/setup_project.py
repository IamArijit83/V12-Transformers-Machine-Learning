import os

print("🚀 Setting up F1 Performance Degradation Project...\n")

# Define folder structure
folders = [
    "data/raw",
    "data/processed",
    "data/models",
    "notebooks",
    "src",
    "results/figures",
    "results/metrics",
    "config"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created: {folder}/")

# Create __init__.py in src
with open("src/__init__.py", "w") as f:
    f.write("# F1 Performance Degradation Package\n")
    f.write("__version__ = '1.0.0'\n")

# Create requirements.txt
requirements = """fastf1>=3.7.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
jupyter>=1.0.0
pyyaml>=6.0
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

# Create README
readme = """# F1 Driver Performance Degradation Prediction

## Project Overview
Predicting driver performance degradation using lap time variance patterns and transformer models.

## Novel Contributions
- First to combine lap variance patterns as biometric proxy
- Transformer architecture treating laps as sequence tokens
- Real F1 telemetry data (2024 season)

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebooks in order:
   - 01_data_exploration.ipynb
   - 02_feature_engineering.ipynb
   - 03_model_training.ipynb

## Hardware Requirements
- GPU: NVIDIA GTX 1650Ti (4GB) or better
- RAM: 8GB minimum
- Storage: 5GB for cached data

## Author
Research Project 2025
"""

with open("README.md", "w") as f:
    f.write(readme)

# Create config file
config = """# Model Configuration
model:
  input_dim: 10
  d_model: 64
  nhead: 4
  num_layers: 2
  dim_feedforward: 256
  dropout: 0.1
  max_seq_length: 50

# Training Configuration
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 20
  sequence_length: 10

# Data Configuration
data:
  year: 2024
  races: [1, 2, 3, 4, 5]
  session_type: 'R'
  cache_dir: 'data/raw'
"""

with open("config/config.yaml", "w") as f:
    f.write(config)

print("\n✅ Project structure created successfully!")
print(f"📁 Location: {os.getcwd()}")
print("\n📋 Next steps:")
print("1. Run: pip install -r requirements.txt")
print("2. Run: jupyter notebook")
print("3. Open notebooks in order (01, 02, 03)")