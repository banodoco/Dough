@echo off
set "folderName=Dough"
if not exist "%folderName%\" (
    if /i not "%CD%"=="%~dp0%folderName%\" (
        git clone --depth 1 -b main https://github.com/banodoco/Dough.git
        cd Dough
        python -m venv dough-env
        call dough-env\Scripts\activate.bat
        python.exe -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install websocket
        call dough-env\Scripts\deactivate.bat
        copy .env.sample .env
        cd ..
        pause
    )
)