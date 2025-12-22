@echo off
REM Torch
python -c "import torch" >nul 2>&1
if %errorlevel%==0 (
    echo Torch found. Reinstalling...
    pip uninstall torch -y
    pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
) else (
    echo Torch not found, skipping installation
)

REM Torchaudio
python -c "import torchaudio" >nul 2>&1
if %errorlevel%==0 (
    echo Torchaudio found. Reinstalling...
    pip uninstall torchaudio -y
    pip install torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
) else (
    echo Torchaudio not found, skipping installation
)

REM Torchvision
python -c "import torchvision" >nul 2>&1
if %errorlevel%==0 (
    echo Torchvision found. Reinstalling...
    pip uninstall torchvision -y
    pip install torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
) else (
    echo Torchvision not found, skipping installation
)

REM Verification
python -c "import torch; print(f'Torch version: {torch.__version__}'); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
