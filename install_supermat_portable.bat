@echo off
chcp 65001 >nul
title Установка SuperMat для Texelith
cd /d "%~dp0"

rem Всё ставится в папку supermat_env рядом с Texelith: свой Python, PyTorch,
rem код SuperMat, его веса. Системный Python не трогается. Повторный запуск
rem безопасен: готовые шаги пропускаются, PyTorch обновляется до нужной версии.

set "SM=%~dp0supermat_env"
set "PY=%SM%\python\python.exe"
if not exist "%SM%" mkdir "%SM%"

echo.
echo === 1/6  Python 3.11 (переносной)
if exist "%PY%" (
    echo уже есть, пропускаю
) else (
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '%SM%\python.zip'" || goto :fail
    powershell -NoProfile -Command "Expand-Archive -Path '%SM%\python.zip' -DestinationPath '%SM%\python' -Force" || goto :fail
    del "%SM%\python.zip"
    powershell -NoProfile -Command "(Get-Content '%SM%\python\python311._pth') -replace '#import site','import site' | Set-Content '%SM%\python\python311._pth'"
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%SM%\get-pip.py'" || goto :fail
    "%PY%" "%SM%\get-pip.py" --no-warn-script-location || goto :fail
    del "%SM%\get-pip.py"
)

echo.
echo === 2/6  Видеокарта и сборка PyTorch
rem RTX 50-й серии (compute capability 12.x) работает только со сборкой cu128;
rem остальным NVIDIA подходит cu126; без NVIDIA — сборка для процессора
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "CC="
for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do if not defined CC set "CC=%%a"
if defined CC (
    if %CC% GEQ 12 (set "TORCH_INDEX=https://download.pytorch.org/whl/cu128") else (set "TORCH_INDEX=https://download.pytorch.org/whl/cu126")
)
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>nul
echo сборка: %TORCH_INDEX%

echo.
echo === 3/6  PyTorch 2.6 или новее
rem 2.5 и старше не подходит: свежие diffusers/transformers требуют torch.accelerator
"%PY%" -m pip install --upgrade "torch>=2.6" torchvision --index-url %TORCH_INDEX% --no-warn-script-location || goto :fail

echo.
echo === 4/6  diffusers, transformers и остальное
rem xformers НЕ ставится: для одиночных текстур SuperMat его не использует,
rem а сам xformers тянет свою версию torch и может её подменить
"%PY%" -m pip install --upgrade diffusers transformers accelerate huggingface_hub safetensors numpy pillow --no-warn-script-location || goto :fail

echo.
echo === 5/6  Код SuperMat и веса
if exist "%SM%\repo\src\adapters.py" (
    echo код уже есть, пропускаю
) else (
    if exist "%SM%\repo" rmdir /s /q "%SM%\repo"
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest 'https://github.com/hyj542682306/SuperMat/archive/refs/heads/main.zip' -OutFile '%SM%\supermat.zip'" || goto :fail
    powershell -NoProfile -Command "Expand-Archive -Path '%SM%\supermat.zip' -DestinationPath '%SM%' -Force" || goto :fail
    move "%SM%\SuperMat-main" "%SM%\repo" >nul || goto :fail
    del "%SM%\supermat.zip"
)
if exist "%SM%\repo\checkpoints\supermat.pth" (
    echo веса уже есть, пропускаю
) else (
    "%PY%" -c "from huggingface_hub import hf_hub_download as d; d('oyiya/SuperMat', 'supermat.pth', local_dir=r'%SM%\repo\checkpoints')" || goto :fail
)

rem пути для Texelith: %USERPROFILE%\.texelith\supermat.json
"%PY%" -c "import json, os; d = os.path.join(os.path.expanduser('~'), '.texelith'); os.makedirs(d, exist_ok=True); json.dump({'repo': r'%SM%\repo', 'python': r'%PY%', 'weights': ''}, open(os.path.join(d, 'supermat.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)" || goto :fail

echo.
echo === 6/6  Проверка: загрузка модели
echo При первом запуске скачивается базовая модель Stable Diffusion 2.1, около 5 ГБ.
"%PY%" -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'processor')"
"%PY%" "%~dp0supermat_worker.py" "%SM%\repo" < nul
if errorlevel 1 goto :fail

echo.
echo ГОТОВО. Строка "@@{"ready": true ...}" выше значит, что SuperMat загрузился.
echo Запускайте Texelith; узел «SuperMat (PBR)» и шаблон «SuperMat: PBR - OpenMW» готовы.
pause
exit /b 0

:fail
echo.
echo ОШИБКА на шаге выше. Сфотографируйте или скопируйте красный текст и пришлите.
pause
exit /b 1
