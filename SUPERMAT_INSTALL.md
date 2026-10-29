# SuperMat для Texelith

SuperMat (код https://github.com/hyj542682306/SuperMat, веса https://huggingface.co/oyiya/SuperMat, лицензия MIT) даёт базовый цвет, шероховатость и металличность. Нормали и высоты он не даёт — их в шаблоне «SuperMat: PBR → OpenMW» берут DeepBump и сеть глубины.

## Установка

Запустите `install_supermat_portable.bat` рядом с Texelith. Он:

1. ставит переносной Python 3.11 в `supermat_env\python` (системный Python не трогается);
2. по видеокарте выбирает сборку PyTorch: RTX 50-й серии — cu128, другие NVIDIA — cu126, без NVIDIA — процессор;
3. ставит PyTorch 2.6 или новее (2.5 не подходит), diffusers, transformers, accelerate — без xformers;
4. скачивает код SuperMat в `supermat_env\repo` и веса `supermat.pth` в `supermat_env\repo\checkpoints`;
5. записывает пути в `%USERPROFILE%\.texelith\supermat.json`;
6. проверяет загрузку модели. В первый раз скачивается базовая модель Stable Diffusion 2.1 (около 5 ГБ, лицензия CreativeML Open RAIL++-M).

Повторный запуск безопасен: готовое пропускается, PyTorch обновляется.

Остальные `install_supermat_*.bat` и `extract_supermat.bat` — старые попытки, их можно удалить.
