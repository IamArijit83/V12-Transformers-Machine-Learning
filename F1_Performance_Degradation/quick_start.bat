@echo off
echo ========================================
echo F1 Performance Degradation Project
echo ========================================
echo.

cd /d "D:\Research Projects\F1_Performance_Degradation"

call conda activate f1_degradation

echo Environment activated: f1_degradation
echo Current directory: %cd%
echo.
echo Options:
echo 1. Launch Jupyter Notebook
echo 2. Run data loader test
echo 3. Check GPU status
echo 4. Exit
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    jupyter notebook
) else if "%choice%"=="2" (
    python src/data_loader.py
) else if "%choice%"=="3" (
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    pause
) else (
    exit
)