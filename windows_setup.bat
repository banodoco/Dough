@echo off
set "folderName=Dough"
for %%I in ("%~dp0.") do set ParentFolderName=%%~nxI
if not exist "%folderName%\" (
    if not "%folderName%"=="%ParentFolderName%" (
        git clone --depth 1 -b main https://github.com/banodoco/Dough.git
        cd Dough
        git clone --depth 1 -b feature/package https://github.com/piyushK52/comfy_runner.git
        git clone https://github.com/comfyanonymous/ComfyUI.git
        python -m venv dough-env
        call dough-env\Scripts\activate.bat
        python.exe -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install websocket
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install -r comfy_runner\requirements.txt
        pip install -r ComfyUI\requirements.txt
	    powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl', 'insightface-0.7.3-cp310-cp310-win_amd64.whl')"
        pip install insightface-0.7.3-cp310-cp310-win_amd64.whl
        del insightface-0.7.3-cp310-cp310-win_amd64.whl
        call dough-env\Scripts\deactivate.bat
        copy .env.sample .env
        cd ..
        pause
    )
)