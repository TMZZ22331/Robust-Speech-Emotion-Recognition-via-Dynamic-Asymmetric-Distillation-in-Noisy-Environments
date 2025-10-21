param(
    [Parameter(Mandatory=$true)]
    [string]$EMODB_ROOT,
    
    [Parameter(Mandatory=$false)]
    [string]$output_base_path = "C:\Users\admin\Desktop\DATA",
    
    [Parameter(Mandatory=$false)]
    [string]$noise_root = "C:\Users\admin\Desktop\DATA\noise-92\NOISE\data\signals\5types",
    
    [Parameter(Mandatory=$false)]
    [array]$snr_levels = @(0, 5, 10, 15, 20),
    
    [Parameter(Mandatory=$false)]
    [array]$noise_types = @()
)
# .\real_noise_preprocessing.ps1 -EMODB_ROOT "C:\Users\admin\Desktop\DATA\EmoDB\EmoDB_Dataset_wav_datasets" -output_base_path "C:\Users\admin\Desktop\DATA\processed_features_EMODB_noisy" -snr_levels @(20,15,10,0)

# 解决Python脚本在GBK环境下打印Unicode字符（如emoji）时的编码错误
$env:PYTHONIOENCODING = "utf-8"

# 添加噪声类型自动检测函数 - EMODB版本（去除F16噪声）
function Get-AvailableNoiseTypes {
    param([string]$NoiseRoot)
    
    Write-Host "正在检测可用的噪声类型（EMODB版本，已去除F16）..." -ForegroundColor Yellow
    
    if (!(Test-Path $NoiseRoot)) {
        Write-Error "噪声目录不存在: $NoiseRoot"
        return @()
    }
    
    $wavFiles = Get-ChildItem "$NoiseRoot\*.wav" -ErrorAction SilentlyContinue
    if ($wavFiles.Count -eq 0) {
        Write-Error "在噪声目录中未找到.wav文件: $NoiseRoot"
        return @()
    }
    
    $detectedTypes = @{}
    
    foreach ($file in $wavFiles) {
        $filename = $file.Name.ToLower()
        $noiseType = ""
        
        # 先去掉扩展名
        $nameWithoutExt = [System.IO.Path]::GetFileNameWithoutExtension($filename)
        
        # 根据标准4types文件精确匹配（去除F16）
        if ($nameWithoutExt -eq "babble") {
            $noiseType = "babble"
        } elseif ($nameWithoutExt -eq "factory1") {
            $noiseType = "factory"
        } elseif ($nameWithoutExt -eq "hfchannel") {
            $noiseType = "hfchannel"
        } elseif ($nameWithoutExt -eq "volvo") {
            $noiseType = "volvo"
        } elseif ($filename -match "factory") {
            $noiseType = "factory"
        } elseif ($filename -match "car|vehicle") {
            $noiseType = "car"
        } elseif ($filename -match "street|traffic") {
            $noiseType = "street"
        } elseif ($filename -match "cafe|restaurant") {
            $noiseType = "cafe"
        } elseif ($filename -match "office|meeting") {
            $noiseType = "office"
        } else {
            # 跳过F16及其他不支持的噪声类型
            if ($nameWithoutExt -ne "f16") {
                # 如果无法分类且不是F16，使用去掉扩展名的文件名第一部分
                $parts = $nameWithoutExt -split "_|-"
                if ($parts.Length -gt 0) {
                    $firstPart = $parts[0]
                    if ($firstPart -ne "f16") {
                        $noiseType = $firstPart
                    }
                }
            }
        }
        
        if ($noiseType -ne "" -and -not $detectedTypes.ContainsKey($noiseType)) {
            $detectedTypes[$noiseType] = @()
        }
        if ($noiseType -ne "") {
            $detectedTypes[$noiseType] += $file.Name
        }
    }
    
    Write-Host "检测到的噪声类型（已排除F16）:" -ForegroundColor Cyan
    foreach ($type in $detectedTypes.Keys) {
        $fileCount = $detectedTypes[$type].Count
        Write-Host "  - ${type}: $fileCount 个文件" -ForegroundColor Green
    }
    
    return $detectedTypes.Keys
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "EMODB Real Noise Preprocessing Pipeline (No F16)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 检查输入参数
if (!(Test-Path $EMODB_ROOT)) {
    Write-Error "EMODB root directory not found: $EMODB_ROOT"
    exit 1
}

if (!(Test-Path $noise_root)) {
    Write-Error "Noise root directory not found: $noise_root"
    exit 1
}

# 自动检测噪声类型（如果没有手动指定）
if ($noise_types.Count -eq 0) {
    $noise_types = Get-AvailableNoiseTypes -NoiseRoot $noise_root
    if ($noise_types.Count -eq 0) {
        Write-Error "未检测到可用的噪声类型，请检查噪声文件夹"
        exit 1
    }
}

Write-Host "Input EMODB Root: $EMODB_ROOT" -ForegroundColor Cyan
Write-Host "Noise Root: $noise_root" -ForegroundColor Cyan
Write-Host "Output Base Directory: $output_base_path" -ForegroundColor Cyan
$snr_str = $snr_levels -join ', '
Write-Host "SNR Levels: $snr_str dB" -ForegroundColor Cyan
$noise_str = $noise_types -join ', '
Write-Host "Detected Noise Types: $noise_str (F16已排除)" -ForegroundColor Cyan
Write-Host ""

$global_start_time = Get-Date
$processed_count = 0
$failed_count = 0

# 按噪声类型分别处理
Write-Host "========================================" -ForegroundColor Blue
Write-Host "按噪声类型分别处理EMODB数据" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

foreach ($noise_type in $noise_types) {
    Write-Host "Processing noise type: $noise_type" -ForegroundColor Yellow
    
    foreach ($snr_db in $snr_levels) {
        $type_start_time = Get-Date
        $output_path = "$output_base_path\root1-$noise_type-${snr_db}db"
        
        Write-Host "  Processing $noise_type at ${snr_db}dB..." -ForegroundColor White
        Write-Host "  Output: $output_path" -ForegroundColor Gray
        
        try {
            # 创建输出目录
            if (!(Test-Path $output_path)) {
                New-Item -ItemType Directory -Path $output_path -Force | Out-Null
            } else {
                Remove-Item "$output_path\*" -Force -Recurse -ErrorAction SilentlyContinue
            }

            # 创建临时目录用于存储噪声音频
            $noisyAudioPath = "$output_path\noisy_audio_temp"
            New-Item -ItemType Directory -Path $noisyAudioPath -Force | Out-Null

            # 步骤1: 生成EMODB标签和manifest文件
            Write-Host "    Step 1: Generating EMODB labels and manifest..." -ForegroundColor Green

            # 生成初始manifest文件
            $pythonScript = "scripts/emodb_manifest.py"
            if (Test-Path $pythonScript) {
                & python $pythonScript --root $EMODB_ROOT --dest $output_path
            } else {
                Write-Error "Python script not found: $pythonScript"
                continue
            }

            # 步骤2: 添加指定类型的真实噪声
            Write-Host "    Step 2: Adding $noise_type noise (SNR=${snr_db}dB)..." -ForegroundColor Green

            $addNoiseScript = "scripts/add_real_noise_to_audio.py"
            if (Test-Path $addNoiseScript) {
                # 映射噪声类型名称到标准名称（去除F16）
                $standardNoiseType = $noise_type
                switch ($noise_type.ToLower()) {
                    "babble" { $standardNoiseType = "babble" }
                    "factory" { $standardNoiseType = "factory" }
                    "factory1" { $standardNoiseType = "factory" }
                    "hfchannel" { $standardNoiseType = "hfchannel" }
                    "volvo" { $standardNoiseType = "volvo" }
                    default { 
                        Write-Warning "未知的噪声类型: $noise_type，将尝试直接使用"
                        $standardNoiseType = $noise_type
                    }
                }
                
                Write-Host "    使用标准噪声类型: $standardNoiseType" -ForegroundColor Cyan
                
                $arguments = @(
                    $addNoiseScript,
                    "--input_root", $EMODB_ROOT,
                    "--output_root", $noisyAudioPath,
                    "--noise_root", $noise_root,
                    "--snr_db", $snr_db,
                    "--manifest_path", "$output_path\train.tsv",
                    "--noise_mode", "type_specific",
                    "--noise_type", $standardNoiseType
                )
                
                $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
                if ($process.ExitCode -ne 0) {
                    Write-Error "Failed to add $noise_type noise. Exit code: $($process.ExitCode)"
                    continue
                }
            } else {
                Write-Error "Noise addition script not found: $addNoiseScript"
                continue
            }

            # 步骤3: 验证噪声注入效果
            Write-Host "    Step 3: Verifying noise injection..." -ForegroundColor Green

            $verifyNoiseScript = "scripts/verify_noise_injection.py"
            if (Test-Path $verifyNoiseScript) {
                $arguments = @(
                    $verifyNoiseScript,
                    "--clean_root", $EMODB_ROOT,
                    "--noisy_root", $noisyAudioPath,
                    "--expected_snr", $snr_db,
                    "--noise_type", $standardNoiseType,
                    "--sample_count", "15",
                    "--tolerance", "3.0"
                )
                
                $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
                if ($process.ExitCode -ne 0) {
                    Write-Error "[ERROR] 噪声注入验证失败! 噪声没有被正确添加到音频中。"
                    Write-Host "    检查项目:" -ForegroundColor Yellow
                    Write-Host "      1. 噪声文件是否正确加载: $noise_root" -ForegroundColor White
                    Write-Host "      2. SNR设置是否正确: ${snr_db}dB" -ForegroundColor White
                    Write-Host "      3. 噪声类型是否匹配: $standardNoiseType" -ForegroundColor White
                    Write-Host "      4. 音频处理管道是否正常" -ForegroundColor White
                    Write-Host "    [STOP] 停止处理以避免生成无效数据。" -ForegroundColor Red
                    continue
                } else {
                    Write-Host "    [OK] 噪声注入验证通过! $standardNoiseType 噪声已成功添加 (SNR=${snr_db}dB)" -ForegroundColor Green
                }
            } else {
                Write-Warning "噪声验证脚本不存在，跳过验证: $verifyNoiseScript"
            }

            # 步骤4: 重新生成Manifest文件
            Write-Host "    Step 4: Regenerating manifest for noisy audio..." -ForegroundColor Green

            $noisyManifestScript = "scripts/emodb_manifest_noisy.py"
            if (Test-Path $noisyManifestScript) {
                & python $noisyManifestScript --root $noisyAudioPath --dest $output_path --original-manifest-dir $output_path
            } else {
                Write-Error "Noisy manifest script not found: $noisyManifestScript"
                continue
            }

            # 步骤5: 检查和修复音频格式
            Write-Host "    Step 5: Checking and fixing audio format..." -ForegroundColor Green

            $audioFixScript = "scripts/check_and_fix_audio_format.py"
            if (Test-Path $audioFixScript) {
                $manifestPath = "$output_path\train.tsv"
                $arguments = @(
                    $audioFixScript,
                    "--mode", "manifest",
                    "--input", $manifestPath
                )
                
                $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
                if ($process.ExitCode -ne 0) {
                    Write-Warning "Audio format check found issues, attempting to fix..."
                    
                    # 尝试修复音频格式
                    $arguments = @(
                        $audioFixScript,
                        "--mode", "fix",
                        "--input", $manifestPath
                    )
                    
                    $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
                    if ($process.ExitCode -ne 0) {
                        Write-Error "Failed to fix audio format issues"
                        continue
                    }
                }
            } else {
                Write-Warning "Audio format check script not found: $audioFixScript"
            }

            # 步骤6: 特征提取
            Write-Host "    Step 6: Extracting features from noisy audio..." -ForegroundColor Green

            # 设置环境变量
            $env:CUDA_LAUNCH_BLOCKING = "1"
            $env:HYDRA_FULL_ERROR = "1"
            $env:CUDA_VISIBLE_DEVICES = "0"
            $env:PYTHONPATH = ".."

            $checkpointPath = "emotion2vec_base.pt"
            $modelPath = "upstream"

            if (!(Test-Path $checkpointPath)) {
                Write-Error "Checkpoint file not found: $checkpointPath"
                continue
            }

            if (!(Test-Path $modelPath)) {
                Write-Error "Upstream directory not found: $modelPath"
                continue
            }

            $arguments = @(
                "scripts/emotion2vec_speech_features.py",
                "--data", $output_path,
                "--model", $modelPath,
                "--split=train",
                "--checkpoint=$checkpointPath",
                "--save-dir=$output_path",
                "--layer=11"
            )

            $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
            if ($process.ExitCode -ne 0) {
                Write-Error "Feature extraction failed with exit code: $($process.ExitCode)"
                Write-Host "    尝试的解决方案:" -ForegroundColor Yellow
                Write-Host "      1. 检查音频文件是否为16kHz单声道格式" -ForegroundColor White
                Write-Host "      2. 检查CUDA环境是否正常" -ForegroundColor White
                Write-Host "      3. 检查emotion2vec模型文件是否完整" -ForegroundColor White
                continue
            }

            # 步骤7: 清理临时文件
            Write-Host "    Step 7: Cleaning up temporary files..." -ForegroundColor Green
            # Remove-Item $noisyAudioPath -Force -Recurse -ErrorAction SilentlyContinue

            $type_end_time = Get-Date
            $type_duration = $type_end_time - $type_start_time
            $type_duration_minutes = $type_duration.TotalMinutes.ToString('0.1')
            Write-Host "    Completed: $noise_type at ${snr_db}dB in $type_duration_minutes minutes" -ForegroundColor Green
            
            $processed_count++
            
        } catch {
            $error_msg = $_.Exception.Message
            Write-Error "    Error processing $noise_type at ${snr_db}dB: $error_msg"
            $failed_count++
            continue
        }
    }
    Write-Host ""
}

# 全局完成总结
$global_end_time = Get-Date
$global_duration = $global_end_time - $global_start_time

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "EMODB Real Noise Preprocessing Pipeline Completed! (No F16)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host ""
Write-Host "Processing Summary:" -ForegroundColor Cyan
$total_expected = ($noise_types.Count * $snr_levels.Count)
Write-Host "  Total expected datasets: $total_expected (按噪声类型分别处理，F16已排除)" -ForegroundColor White
Write-Host "  Successfully processed: $processed_count" -ForegroundColor Green
Write-Host "  Failed: $failed_count" -ForegroundColor Red
$global_duration_minutes = $global_duration.TotalMinutes.ToString('0.1')
Write-Host "  Total duration: $global_duration_minutes minutes" -ForegroundColor White

Write-Host ""
Write-Host "EMODB real noise preprocessing pipeline completed! (F16噪声已去除)" -ForegroundColor Green