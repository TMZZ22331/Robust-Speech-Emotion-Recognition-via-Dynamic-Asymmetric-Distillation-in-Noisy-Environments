<#
.SYNOPSIS
    用于EmoDB情感语料库的统一预处理流水线。
    支持干净数据和带噪数据的特征提取。

.DESCRIPTION
    该脚本完成以下任务:
    1. 调用 emodb_manifest.py 为原始音频生成 manifest, label, 和 speaker 文件。
    2. (可选) 如果指定了 -AddNoise, 则会为每个SNR级别循环执行以下操作:
        a. 创建一个临时目录，并使用 add_noise_to_audio.py 生成带噪声音频。
        b. 调用 emodb_manifest_noisy.py 为带噪声音频生成新的 manifest。
    3. 调用 emotion2vec_speech_features.py 提取特征。
    4. 清理临时文件。

.PARAMETER EMODB_ROOT
    EmoDB情感语料库的根目录的绝对路径。
    例如: "C:\Users\admin\Desktop\DATA\EmoDB\EmoDB Dataset_wav_datasets"

.PARAMETER OutputBasePath
    用于存放所有输出文件的基础目录。
    脚本会在此目录下为每个处理任务创建子目录。
    例如: "C:\Users\admin\Desktop\DATA\EmoDB_processed"

.PARAMETER AddNoise
    一个开关参数。如果提供此参数，脚本将执行带噪数据处理流程。
    否则，将只处理干净数据。

.PARAMETER SnrLevels
    一个整数数组，定义了在添加噪声时要使用的信噪比 (SNR) 级别 (dB)。
    仅在提供了 -AddNoise 开关时有效。

.EXAMPLE
    # 示例1: 只处理干净数据
    .\emodb_preprocessing.ps1 -EMODB_ROOT "C:\Users\admin\Desktop\DATA\EmoDB\EmoDB Dataset_wav_datasets" -OutputBasePath "C:\Users\admin\Desktop\DATA\processed_features_EMODB"

.EXAMPLE
    # 示例2: 处理多种信噪比的噪声数据 (0dB, 5dB, 10dB)
    .\emodb_preprocessing.ps1 -EMODB_ROOT "C:\Users\admin\Desktop\DATA\EmoDB\EmoDB Dataset_wav_datasets" -OutputBasePath "C:\Users\admin\Desktop\DATA\meihua\EMODB" -AddNoise -SnrLevels @(20)
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$EMODB_ROOT,
    
    [Parameter(Mandatory=$true)]
    [string]$OutputBasePath,

    [Parameter(Mandatory=$false)]
    [switch]$AddNoise,

    [Parameter(Mandatory=$false)]
    [array]$SnrLevels = @(0, 5, 10, 15, 20)
)

# --- 全局设置 ---
$global_start_time = Get-Date
$python_executable = "python" # 或者 "python3"
$base_script_path = $PSScriptRoot

# --- 脚本路径检查 ---
$emodb_manifest_script = Join-Path $base_script_path "scripts\emodb_manifest.py"
$emodb_manifest_noisy_script = Join-Path $base_script_path "scripts\emodb_manifest_noisy.py"
$add_noise_script = Join-Path $base_script_path "scripts\add_noise_to_audio.py"
$feature_extraction_script = Join-Path $base_script_path "scripts\emotion2vec_speech_features.py"
$checkpoint_file = Join-Path $base_script_path "emotion2vec_base.pt"

$required_scripts = @(
    $emodb_manifest_script, $emodb_manifest_noisy_script, 
    $add_noise_script, $feature_extraction_script, $checkpoint_file
)
foreach ($script in $required_scripts) {
    if (!(Test-Path $script)) {
        Write-Error "关键文件未找到: $script"
        exit 1
    }
}

# --- 函数定义 ---
function Run-Feature-Extraction {
    param(
        [string]$DataPath,
        [string]$SaveDir
    )
    
    Write-Host ""
    Write-Host "Step: Extracting features from emotion2vec model..." -ForegroundColor Green
    Write-Host "Data source: $DataPath"
    Write-Host "This may take a while depending on your GPU and dataset size..." -ForegroundColor Yellow

    # 设置环境变量
    $env:CUDA_LAUNCH_BLOCKING = "1"
    $env:HYDRA_FULL_ERROR = "1"
    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:PYTHONPATH = $base_script_path # 确保 upstream 可以被找到

    $arguments = @(
        $feature_extraction_script,
        "--data", $DataPath,
        "--model", "upstream",
        "--split=train",
        "--checkpoint=$checkpoint_file",
        "--save-dir=$SaveDir",
        "--layer=11"
    )

    try {
        # 切换到脚本所在目录运行
        $original_location = Get-Location
        Set-Location $base_script_path
        
        $process = Start-Process -FilePath $python_executable -ArgumentList $arguments -Wait -PassThru -NoNewWindow
        
        # 恢复原始位置
        Set-Location $original_location
        
        if ($process.ExitCode -eq 0) {
            Write-Host "Feature extraction completed successfully!" -ForegroundColor Green
        } else {
            throw "Feature extraction failed with exit code: $($process.ExitCode)"
        }
    } catch {
        # 确保恢复原始位置
        Set-Location $original_location
        throw "Error during feature extraction: $_"
    }
}


# --- 主流程 ---

if (!$AddNoise) {
    # --- 流程1: 处理干净数据 ---
    $output_path = Join-Path $OutputBasePath "processed_features_clean"
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "EmoDB Clean Data Preprocessing Pipeline" -ForegroundColor Cyan
    Write-Host "Output: $output_path" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    # 创建输出目录
    New-Item -ItemType Directory -Path $output_path -Force | Out-Null
    
    # 步骤1: 生成Manifest
    Write-Host "Step 1: Generating manifest for clean audio..." -ForegroundColor Green
    python $emodb_manifest_script --root $EMODB_ROOT --dest $output_path
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to generate manifest for clean audio."; exit 1 }

    # 步骤2: 特征提取
    Run-Feature-Extraction -DataPath $output_path -SaveDir $output_path

} else {
    # --- 流程2: 处理带噪数据 ---
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "EmoDB Noisy Data Preprocessing Pipeline" -ForegroundColor Yellow
    Write-Host "SNR Levels: $($SnrLevels -join ', ') dB" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow

    # 步骤1: (仅一次) 生成干净数据的Manifest，作为后续步骤的基础
    $original_manifest_path = Join-Path $OutputBasePath "temp_manifest_clean"
    New-Item -ItemType Directory -Path $original_manifest_path -Force | Out-Null
    Write-Host "Step 1: Generating base manifest from clean audio..." -ForegroundColor Green
    python $emodb_manifest_script --root $EMODB_ROOT --dest $original_manifest_path
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to generate base manifest."; exit 1 }

    foreach ($snr_db in $SnrLevels) {
        $snr_start_time = Get-Date
        $output_path_snr = Join-Path $OutputBasePath "processed_features_noisy_${snr_db}db"
        
        Write-Host "`n"
        Write-Host "--------------------------------------------------" -ForegroundColor Yellow
        Write-Host "Processing SNR: $snr_db dB" -ForegroundColor Yellow
        Write-Host "Output: $output_path_snr" -ForegroundColor Yellow
        Write-Host "--------------------------------------------------"
        
        # 创建当前SNR的输出目录
        New-Item -ItemType Directory -Path $output_path_snr -Force | Out-Null

        # 步骤 2.1: 添加噪声
        Write-Host "Step 2.1: Adding white noise (SNR=$snr_db dB)..." -ForegroundColor Green
        $noisy_audio_temp_path = Join-Path $output_path_snr "noisy_audio_temp"
        New-Item -ItemType Directory -Path $noisy_audio_temp_path -Force | Out-Null
        
        $noise_args = @(
            $add_noise_script,
            "--input_root", $EMODB_ROOT,
            "--output_root", $noisy_audio_temp_path,
            "--snr_db", $snr_db,
            # 使用最开始生成的干净manifest
            "--manifest_path", (Join-Path $original_manifest_path "train.tsv")
        )
        python $noise_args
        if ($LASTEXITCODE -ne 0) { Write-Error "Failed to add noise for SNR $snr_db."; continue }

        # 步骤 2.2: 为噪声音频生成新Manifest
        Write-Host "Step 2.2: Regenerating manifest for noisy audio..." -ForegroundColor Green
        $noisy_manifest_args = @(
            $emodb_manifest_noisy_script,
            "--root", $noisy_audio_temp_path,
            "--original-manifest-dir", $original_manifest_path,
            "--dest", $output_path_snr
        )
        python $noisy_manifest_args
        if ($LASTEXITCODE -ne 0) { Write-Error "Failed to generate noisy manifest for SNR $snr_db."; continue }

        # 步骤 2.3: 提取特征
        Run-Feature-Extraction -DataPath $output_path_snr -SaveDir $output_path_snr

        # 步骤 2.4: 清理临时噪声音频
        Write-Host "Step 2.4: Cleaning up temporary noisy audio files..." -ForegroundColor Green
        Remove-Item $noisy_audio_temp_path -Force -Recurse -ErrorAction SilentlyContinue

        $snr_duration = (Get-Date) - $snr_start_time
        Write-Host "✅ SNR $snr_db dB processing completed in $($snr_duration.TotalSeconds.ToString('F1')) seconds." -ForegroundColor Green
    }

    # 清理最开始生成的干净manifest
    Write-Host "`nCleaning up temporary base manifest..." -ForegroundColor Green
    Remove-Item $original_manifest_path -Force -Recurse -ErrorAction SilentlyContinue
}

$global_duration = (Get-Date) - $global_start_time
Write-Host "`n"
Write-Host "========================================" -ForegroundColor Green
Write-Host "EmoDB Preprocessing Pipeline Completed!" -ForegroundColor Green
Write-Host "Total Duration: $($global_duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
Write-Host "Generated files are in: $OutputBasePath" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green 