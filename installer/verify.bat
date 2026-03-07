@echo off
echo.
echo   Verifying theSpaceDB installation...
echo   ═════════════════════════════════════
echo.

SET PASS=0
SET FAIL=0

:: Check 1 — Install directory
IF EXIST "%LOCALAPPDATA%\theSpaceDB\" (
    echo   [PASS] Install directory found: %LOCALAPPDATA%\theSpaceDB
    SET /A PASS+=1
) ELSE (
    echo   [FAIL] Install directory not found
    SET /A FAIL+=1
)

:: Check 2 — Virtual environment
IF EXIST "%LOCALAPPDATA%\theSpaceDB\.venv\Scripts\python.exe" (
    echo   [PASS] Virtual environment found
    SET /A PASS+=1
) ELSE (
    echo   [FAIL] Virtual environment missing
    SET /A FAIL+=1
)

:: Check 3 — spacesh launcher
IF EXIST "%LOCALAPPDATA%\theSpaceDB\spacesh.bat" (
    echo   [PASS] spacesh launcher found
    SET /A PASS+=1
) ELSE (
    echo   [FAIL] spacesh launcher missing
    SET /A FAIL+=1
)

:: Check 4 — spacedb package importable
"%LOCALAPPDATA%\theSpaceDB\.venv\Scripts\python.exe" -c "import spacedb; print('   [PASS] spacedb package importable  v' + spacedb.__version__)" 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo   [FAIL] spacedb package not importable
    SET /A FAIL+=1
) ELSE (
    SET /A PASS+=1
)

:: Check 5 — sentence-transformers
"%LOCALAPPDATA%\theSpaceDB\.venv\Scripts\python.exe" -c "import sentence_transformers; print('   [PASS] sentence-transformers installed')" 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo   [FAIL] sentence-transformers not installed
    SET /A FAIL+=1
) ELSE (
    SET /A PASS+=1
)

:: Check 6 — spacesh in PATH
WHERE spacesh >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo   [PASS] spacesh command found in PATH
    SET /A PASS+=1
) ELSE (
    echo   [WARN] spacesh not in PATH yet — restart terminal and try again
)

:: Check 7 — desktop shortcut
IF EXIST "%USERPROFILE%\Desktop\spacesh.lnk" (
    echo   [PASS] Desktop shortcut found
    SET /A PASS+=1
) ELSE (
    echo   [WARN] Desktop shortcut not found
)

echo.
echo   ═════════════════════════════════════
echo   Result:  %PASS% passed  /  %FAIL% failed
echo.

IF %FAIL% EQU 0 (
    echo   All checks passed! Run: spacesh D:\my_mind
) ELSE (
    echo   Some checks failed. Try running install.bat again.
)

echo.
pause
