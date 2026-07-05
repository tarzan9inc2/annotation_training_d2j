@echo off
chcp 65001 >nul
echo ========================================
echo   Annotation Tool - Setup Script
echo ========================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Detect Python command
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    echo Python found: python
) else (
    py --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=py
        echo Python found: py
    ) else (
        echo Error: Python is not installed or not in PATH
        echo Please install Python 3.10 from python.org
        echo Download: https://www.python.org/downloads/release/python-31011/
        pause
        exit /b 1
    )
)

echo.
%PYTHON_CMD% --version
echo.

REM Remove existing venv if it exists
if exist "%SCRIPT_DIR%venv\" (
    echo Removing existing virtual environment...
    rmdir /s /q "%SCRIPT_DIR%venv"
)

REM Create new virtual environment
echo Creating virtual environment...
%PYTHON_CMD% -m venv "%SCRIPT_DIR%venv"
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call "%SCRIPT_DIR%venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Ask user about CUDA support
echo.
echo Do you want to install CUDA-enabled PyTorch for GPU acceleration?
echo (Recommended for NVIDIA RTX GPUs)
echo.
choice /C YN /M "Install CUDA support"
if errorlevel 2 goto :install_cpu
if errorlevel 1 goto :install_cuda

:install_cuda
echo.
echo Installing CUDA-enabled dependencies (this may take several minutes)...
echo.
python -m pip install -r "%SCRIPT_DIR%requirements_cuda.txt"
if errorlevel 1 (
    echo Error: Failed to install CUDA dependencies
    echo Falling back to CPU version...
    goto :install_cpu
)
echo.
echo Verifying CUDA installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
goto :install_complete

:install_cpu
echo.
echo Installing CPU-only dependencies (this may take several minutes)...
echo.
python -m pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

:install_complete

echo.
echo ========================================
echo   Setup completed successfully!
echo ========================================
echo.
echo You can now run the application using run.bat
echo.
pause
