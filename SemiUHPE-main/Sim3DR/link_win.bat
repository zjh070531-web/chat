@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "C:\Users\Administrator\Desktop\SemiUHPE-main\Sim3DR"
link.exe /nologo /INCREMENTAL:NO /LTCG /DLL ^
  /LIBPATH:D:\ai\miniconda3\libs /LIBPATH:D:\ai\miniconda3 /LIBPATH:D:\ai\miniconda3\PCbuild\amd64 ^
  "/LIBPATH:C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64" ^
  "/LIBPATH:C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\ucrt\x64" ^
  "/LIBPATH:C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\um\x64" ^
  /EXPORT:PyInit_Sim3DR_Cython ^
  build\temp.win-amd64-3.10\Release\lib\rasterize.obj ^
  build\temp.win-amd64-3.10\Release\lib\rasterize_kernel.obj ^
  /OUT:Sim3DR_Cython.cp310-win_amd64.pyd ^
  /IMPLIB:build\temp.win-amd64-3.10\Release\lib\Sim3DR_Cython.cp310-win_amd64.lib
