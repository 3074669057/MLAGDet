@echo off
chcp 65001 >nul
REM ========================================
REM MLAGDet-review-comment-3-package 上传脚本
REM 在网络正常的环境中双击运行此脚本
REM ========================================

echo.
echo ========================================
echo   MLAGDet-review-comment-3 上传工具
echo ========================================
echo.

REM 检查是否已解压
if exist "MLAGDet-review-comment-3-package" (
    echo 目录已存在，进入目录...
    cd MLAGDet-review-comment-3-package
) else (
    echo 解压文件中...
    powershell -Command "Expand-Archive -Path 'MLAGDet-review-comment-3-package.zip' -DestinationPath '.' -Force"
    cd MLAGDet-review-comment-3-package
)

REM 检查 git 是否可用
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Git，请先安装 Git
    echo 下载地址: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM 检查远程仓库是否已配置
git remote -v | findstr "github.com" >nul 2>&1
if %errorlevel% neq 0 (
    echo 配置远程仓库...
    git remote add origin https://github.com/3074669057/MLAGDet.git
)

echo.
echo 正在推送代码到 GitHub...
echo.

git push -u origin master --force

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   上传成功！
    echo   请访问 https://github.com/3074669057/MLAGDet 查看
    echo ========================================
) else (
    echo.
    echo ========================================
    echo   上传失败，请检查网络连接
    echo ========================================
)

echo.
pause
