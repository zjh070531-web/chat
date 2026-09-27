@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%"
cd /d "C:\Users\Administrator\Desktop\SemiUHPE-main\Sim3DR"
python setup.py build_ext --inplace
