# Environment Setup for Project

This project requires two separate environments for development and production. Below are the instructions for setting them up.

---

## Development Environment (Pytorch_env)

This environment is used for training, testing, and evaluating machine learning models, and as such has more requirements than it does to simply use the finished application.

### Installation

1. Install Conda if not already installed: [Conda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create the environment using the provided configuration file:
   ```bash
   conda env create -f environments/pytorch_env.yml
   ```
3. Activate the environment:
   ```bash
   conda activate pytorch_env
   ```
4. Verify the environment is working correctly:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   The output should be `True` if CUDA is correctly installed.

### Key Libraries
- PyTorch (with CUDA support)
- Scikit-learn
- Pandas
- Matplotlib
- Joblib

### Usage
This environment is required to:
- Train machine learning models.
- Evaluate and test models.
- Run all Jupyter notebooks.

---

## Application/Inference Environment

This environment is designed for running the application, and the trained models in inference mode.

### Installation

1. Create the environment using the lightweight `requirements.txt` file:
   ```bash
   python -m venv app_env
   source app_env/bin/activate  # For Linux/macOS
   app_env\Scripts\activate     # For Windows
   pip install -r requirements.txt
   ```
2. Verify installation:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

### Key Libraries
- PyTorch (CPU or CUDA version, depending on hardware)
- Joblib (for loading models)
- Pandas (for preprocessing)

### Usage
This environment is required to:
- Serve models for inference.
- Run the application backend (e.g., Flask, FastAPI).

---

## Notes
- Keep the `pytorch_env.yml` file updated as dependencies change during development.
- For production, ensure the `requirements.txt` includes only essential libraries for lightweight deployment.

