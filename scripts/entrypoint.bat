@echo off

set COMMAND=streamlit run app.py --runner.fastReruns false --server.port 5500
goto :loop

:compare_versions
    setlocal EnableDelayedExpansion
    set "ver1=%~1"
    set "ver2=%~2"

    set "ver1=!ver1:.= !"
    set "ver2=!ver2:.= !"
    set "arr1="
    set "arr2="

    set "index=0"
    for %%c in (%ver1%) do (
        set "arr1[!index!]=%%c"
        set /a "index+=1"
    )

    set "index=0"
    for %%c in (%ver2%) do (
        set "arr2[!index!]=%%c"
        set /a "index+=1"
    )

    set CURRENT_DIR=%cd%

    for /L %%i in (1,1,3) do (
        set "v1=!arr1[%%i]!"
        set "v2=!arr2[%%i]!"
        if "!v1!" equ "" set "v1=0"
        if "!v2!" equ "" set "v2=0"
        if !v1! gtr !v2! (
            echo You have the latest version
            endlocal & exit /b
        ) else if !v1! lss !v2! (
            echo A newer version is available. Updating...
            git stash
            rem Step 1: Pull from the current branch
            git pull origin "!git rev-parse --abbrev-ref HEAD!"

            rem Step 2: Check if the comfy_runner folder is present
            if exist "!CURRENT_DIR!\comfy_runner" (
                rem Step 3a: If comfy_runner is present, pull from the feature/package branch
                echo comfy_runner folder found. Pulling from feature/package branch.
                cd comfy_runner
                git pull origin feature/package
                cd "!CURRENT_DIR!"
            ) else (
                rem Step 3b: If comfy_runner is not present, clone the repository
                echo comfy_runner folder not found. Cloning repository.
                set REPO_URL=https://github.com/piyushK52/comfy_runner.git
                git clone "!REPO_URL!" "!CURRENT_DIR!\comfy_runner"
            )

            echo !CURRENT_VERSION! > "!CURRENT_DIR!\scripts\app_version.txt"
            endlocal & exit /b
        )
    )
    echo You have the latest version
    endlocal & exit /b

:update_app
    setlocal EnableDelayedExpansion
    set CURRENT_VERSION=
    for /f "delims=" %%i in ('curl -s "https://raw.githubusercontent.com/banodoco/Dough/feature/final/scripts/app_version.txt"') do set CURRENT_VERSION=%%i

    echo %CURRENT_VERSION% | findstr /r "^[^a-zA-Z]*$" >nul
    if %errorlevel% neq 0 (
        echo Invalid version format: %CURRENT_VERSION%. Expected format: X.X.X (e.g., 1.2.3^)
        endlocal & exit /b
    )

    if not "%CURRENT_VERSION%" == "" (
        echo %CURRENT_VERSION%
    ) else (
        set ERR_MSG=Unable to fetch the current version from the remote repository.
        echo %ERR_MSG%
        exit /b
    )

    set CURRENT_DIR=%cd%
    set LOCAL_VERSION=
    for /f "delims=" %%i in ('type "%CURRENT_DIR%\scripts\app_version.txt"') do set LOCAL_VERSION=%%i

    echo Local version %LOCAL_VERSION%
    echo Current version %CURRENT_VERSION%

    call :compare_versions "%LOCAL_VERSION%" "%CURRENT_VERSION%"
    endlocal & exit /b

:loop
    if "%~1" == "--update" (
        call :update_app
        %COMMAND%
        exit /b
    )
    if not "%~1" == "" (
        echo Invalid option: %1 >&2
        exit /b
    )
    shift
    if not "%~1" == "" goto :loop

%COMMAND%