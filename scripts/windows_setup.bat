@echo off
set "folderName=Dough"
if not exist "%folderName%\" (
    if /i not "%CD%"=="%~dp0%folderName%\" (
        git clone --depth 1 -b main https://github.com/banodoco/Dough.git
        cd Dough
        git clone --depth 1 -b main https://github.com/piyushK52/comfy_runner.git
        git clone https://github.com/comfyanonymous/ComfyUI.git
        python -m venv dough-env
        call dough-env\Scripts\activate.bat
        python.exe -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install websocket
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install -r comfy_runner\requirements.txt
        pip install -r ComfyUI\requirements.txt
        call dough-env\Scripts\deactivate.bat
        copy .env.sample .env
        cd ..
        pause
    )
)