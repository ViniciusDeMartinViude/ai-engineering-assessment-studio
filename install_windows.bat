@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem AI Engineering Assessment Studio Windows bootstrapper.
rem Run this file from Command Prompt, PowerShell, or an Anaconda Prompt.

pushd "%~dp0"
if errorlevel 1 goto :fail
set "REPO_ROOT=%CD%"

echo.
echo ================================================
echo AI Engineering Assessment Studio setup
echo ================================================
echo.

set "ENV_NAME="
set /p "ENV_NAME=Enter a Conda environment name [ai-assessment-studio]: "
if not defined ENV_NAME set "ENV_NAME=ai-assessment-studio"

rem Keep the name portable and safe for activation commands.
echo(%ENV_NAME%| %SystemRoot%\System32\findstr.exe /r /x "[A-Za-z0-9][A-Za-z0-9._-]*" >nul
if errorlevel 1 (
    echo.
    echo Invalid environment name.
    echo Use only letters, numbers, dots, underscores, and hyphens.
    goto :fail
)

set "CONDA_CMD="
for /f "delims=" %%G in ('where conda.bat 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%G"
if not defined CONDA_CMD if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"

if not defined CONDA_CMD (
    for %%G in (
        "%USERPROFILE%\anaconda3\condabin\conda.bat"
        "%USERPROFILE%\miniconda3\condabin\conda.bat"
        "%USERPROFILE%\miniforge3\condabin\conda.bat"
        "%USERPROFILE%\mambaforge\condabin\conda.bat"
        "%ProgramData%\anaconda3\condabin\conda.bat"
        "%ProgramData%\miniconda3\condabin\conda.bat"
        "%ProgramData%\miniforge3\condabin\conda.bat"
        "%ProgramData%\mambaforge\condabin\conda.bat"
    ) do if not defined CONDA_CMD if exist "%%~G" set "CONDA_CMD=%%~G"
)

if not defined CONDA_CMD (
    echo.
    echo Conda was not found from this script.
    echo.
    echo Open an Anaconda Prompt, or initialize Conda for the shell you use:
    echo   conda init cmd.exe
    echo   conda init powershell
    echo Then close and reopen the prompt and run this file again.
    goto :fail
)

echo Using Conda command: %CONDA_CMD%
echo Target environment: %ENV_NAME%
echo.

set "ENV_LIST=%TEMP%\ai_assessment_conda_envs_%RANDOM%.txt"
call "%CONDA_CMD%" env list > "%ENV_LIST%" 2>nul
if errorlevel 1 (
    echo Conda could not list environments.
    del /q "%ENV_LIST%" >nul 2>nul
    goto :fail
)

findstr.exe /l /b /c:"%ENV_NAME% " "%ENV_LIST%" >nul
if errorlevel 1 (
    echo Creating Conda environment "%ENV_NAME%" with Python 3.12...
    call "%CONDA_CMD%" create -n "%ENV_NAME%" python=3.12 pip -y
    if errorlevel 1 (
        del /q "%ENV_LIST%" >nul 2>nul
        echo Conda environment creation failed.
        goto :fail
    )
) else (
    echo Reusing existing environment "%ENV_NAME%".
    call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -c "import sys; print('Python ' + sys.version.split()[0]); raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if errorlevel 1 (
        del /q "%ENV_LIST%" >nul 2>nul
        echo The existing environment must use Python 3.12.
        echo Choose another environment name or recreate this environment manually.
        goto :fail
    )
)
del /q "%ENV_LIST%" >nul 2>nul

echo.
echo Installing the GUI dependencies and editable project package...
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -m pip install --upgrade pip
if errorlevel 1 goto :install_fail
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -m pip install -e ".[gui]"
if errorlevel 1 goto :install_fail

echo.
echo Verifying the installation...
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -c "import PySide6, ai_assessment; print('PySide6 and ai_assessment import successfully.')"
if errorlevel 1 goto :install_fail

echo.
echo ================================================
echo Setup complete
echo ================================================
echo.
echo This script cannot activate the parent CMD or PowerShell session.
echo In a new Command Prompt or Anaconda Prompt, run:
echo   call conda activate %ENV_NAME%
echo   python -m ai_assessment.app --headless-check --workspace candidate_workspaces\C014
echo   python -m ai_assessment.app --workspace candidate_workspaces\C014
echo.
echo In PowerShell, run:
echo   conda activate %ENV_NAME%
echo   python -m ai_assessment.app --headless-check --workspace candidate_workspaces\C014
echo   python -m ai_assessment.app --workspace candidate_workspaces\C014
echo.
echo No model weights or datasets were downloaded by this setup script.
popd
exit /b 0

:install_fail
echo.
echo Dependency installation failed. Review the Conda or pip output above.
goto :fail

:fail
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" set "EXIT_CODE=1"
popd >nul 2>nul
exit /b %EXIT_CODE%
