param(
    [Parameter(Mandatory=$true)]
    [string]$IEMOCAP_ROOT,
    
    [Parameter(Mandatory=$false)]
    [string]$output_base_path = "C:\Users\admin\Desktop\DATA",
    
    [Parameter(Mandatory=$false)]
    [string]$noise_root = "C:\Users\admin\Desktop\DATA\noise-92\NOISE\data\signals\5types",
    
    [Parameter(Mandatory=$false)]
    [array]$snr_levels = @(0, 5, 10, 15, 20),
    
    [Parameter(Mandatory=$false)]
    [array]$noise_types = @()
)
# .\real_noise_preprocessing.ps1 -IEMOCAP_ROOT "C:\Users\admin\Desktop\DATA\IEMOCAP_full_release" -output_base_path "C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy" -snr_levels @(20,15,10,0)
# 添加噪声类型自动检测函数
function Get-AvailableNoiseTypes {
    param([string]$NoiseRoot)
    
    Write-Host "正在检测可用的噪声类型..." -ForegroundColor Yellow
    
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
        
        # 根据标准5types文件精确匹配
        if ($nameWithoutExt -eq "babble") {
            $noiseType = "babble"
        } elseif ($nameWithoutExt -eq "f16") {
            $noiseType = "f16"
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
            # 如果无法分类，使用去掉扩展名的文件名第一部分
            $parts = $nameWithoutExt -split "_|-"
            if ($parts.Length -gt 0) {
                $noiseType = $parts[0]
            }
        }
        
        if ($noiseType -ne "" -and -not $detectedTypes.ContainsKey($noiseType)) {
            $detectedTypes[$noiseType] = @()
        }
        if ($noiseType -ne "") {
            $detectedTypes[$noiseType] += $file.Name
        }
    }
    
    Write-Host "检测到的噪声类型:" -ForegroundColor Cyan
    foreach ($type in $detectedTypes.Keys) {
        $fileCount = $detectedTypes[$type].Count
        Write-Host "  - ${type}: $fileCount 个文件" -ForegroundColor Green
    }
    
    return $detectedTypes.Keys
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "IEMOCAP Real Noise Preprocessing Pipeline" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 检查输入参数
if (!(Test-Path $IEMOCAP_ROOT)) {
    Write-Error "IEMOCAP root directory not found: $IEMOCAP_ROOT"
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

Write-Host "Input IEMOCAP Root: $IEMOCAP_ROOT" -ForegroundColor Cyan
Write-Host "Noise Root: $noise_root" -ForegroundColor Cyan
Write-Host "Output Base Directory: $output_base_path" -ForegroundColor Cyan
$snr_str = $snr_levels -join ', '
Write-Host "SNR Levels: $snr_str dB" -ForegroundColor Cyan
$noise_str = $noise_types -join ', '
Write-Host "Detected Noise Types: $noise_str" -ForegroundColor Cyan
Write-Host ""

$global_start_time = Get-Date
$processed_count = 0
$failed_count = 0

# 第一类：按噪声类型分别处理
Write-Host "========================================" -ForegroundColor Blue
Write-Host "第一类：按噪声类型分别处理" -ForegroundColor Blue
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

            # 步骤1: 提取标签和生成初始Manifest
            Write-Host "    Step 1: Extracting emotion labels..." -ForegroundColor Green

            # 处理每个Session的情感标签
            for ($index = 1; $index -le 5; $index++) {
                $sessionPath = Join-Path $IEMOCAP_ROOT "Session$index\dialog\EmoEvaluation"
                $outputFile = Join-Path $output_path "Session$index.emo"
                
                if (Test-Path $sessionPath) {
                    $txtFiles = Get-ChildItem -Path $sessionPath -Filter "*.txt"
                    $allLines = @()
                    
                    foreach ($file in $txtFiles) {
                        $content = Get-Content $file.FullName
                        foreach ($line in $content) {
                            if ($line -match "Ses") {
                                $parts = $line -split "`t"
                                if ($parts.Length -ge 3) {
                                    $col2 = $parts[1].Trim()
                                    $col3 = $parts[2].Trim()
                                    
                                    if ($col3 -eq "ang" -or $col3 -eq "exc" -or $col3 -eq "hap" -or $col3 -eq "neu" -or $col3 -eq "sad") {
                                        if ($col3 -eq "exc") { 
                                            $col3 = "hap" 
                                        }
                                        $allLines += "$col2`t$col3"
                                    }
                                }
                            }
                        }
                    }
                    
                    $allLines | Out-File -FilePath $outputFile -Encoding UTF8
                }
            }

            # 合并所有Session文件到train.emo
            $trainFile = Join-Path $output_path "train.emo"
            $allTrainLines = @()

            for ($index = 1; $index -le 5; $index++) {
                $sessionFile = Join-Path $output_path "Session$index.emo"
                if (Test-Path $sessionFile) {
                    $lines = Get-Content $sessionFile
                    $allTrainLines += $lines
                    Remove-Item $sessionFile
                }
            }

            $allTrainLines | Out-File -FilePath $trainFile -Encoding UTF8

            # 生成初始manifest文件
            $pythonScript = "scripts/iemocap_manifest.py"
            if (Test-Path $pythonScript) {
                python $pythonScript --root $IEMOCAP_ROOT --dest $output_path --label_path $trainFile
            } else {
                Write-Error "Python script not found: $pythonScript"
                continue
            }

            # 步骤2: 添加指定类型的真实噪声
            Write-Host "    Step 2: Adding $noise_type noise (SNR=${snr_db}dB)..." -ForegroundColor Green

            $addNoiseScript = "scripts/add_real_noise_to_audio.py"
            if (Test-Path $addNoiseScript) {
                # 映射噪声类型名称到标准名称
                $standardNoiseType = $noise_type
                switch ($noise_type.ToLower()) {
                    "babble" { $standardNoiseType = "babble" }
                    "f16" { $standardNoiseType = "f16" }
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
                    "--input_root", $IEMOCAP_ROOT,
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
                    "--clean_root", $IEMOCAP_ROOT,
                    "--noisy_root", $noisyAudioPath,
                    "--expected_snr", $snr_db,
                    "--noise_type", $standardNoiseType,
                    "--sample_count", "15",
                    "--tolerance", "3.0"
                )
                
                $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
                if ($process.ExitCode -ne 0) {
                    Write-Error "💥 噪声注入验证失败! 噪声没有被正确添加到音频中。"
                    Write-Host "    检查项目:" -ForegroundColor Yellow
                    Write-Host "      1. 噪声文件是否正确加载: $noise_root" -ForegroundColor White
                    Write-Host "      2. SNR设置是否正确: ${snr_db}dB" -ForegroundColor White
                    Write-Host "      3. 噪声类型是否匹配: $standardNoiseType" -ForegroundColor White
                    Write-Host "      4. 音频处理管道是否正常" -ForegroundColor White
                    Write-Host "    🛑 停止处理以避免生成无效数据。" -ForegroundColor Red
                    continue
                } else {
                    Write-Host "    ✅ 噪声注入验证通过! $standardNoiseType 噪声已成功添加 (SNR=${snr_db}dB)" -ForegroundColor Green
                }
            } else {
                Write-Warning "噪声验证脚本不存在，跳过验证: $verifyNoiseScript"
            }

            # 步骤4: 重新生成Manifest文件
            Write-Host "    Step 4: Regenerating manifest for noisy audio..." -ForegroundColor Green

            $noisyManifestScript = "scripts/iemocap_manifest_noisy.py"
            if (Test-Path $noisyManifestScript) {
                python $noisyManifestScript --root $noisyAudioPath --dest $output_path --label_path $trainFile
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
                $exit_code = $process.ExitCode
                Write-Error "Feature extraction failed with exit code: $exit_code"
                Write-Host "    尝试的解决方案:" -ForegroundColor Yellow
                Write-Host "      1. 检查音频文件是否为16kHz单声道格式" -ForegroundColor White
                Write-Host "      2. 检查CUDA环境是否正常" -ForegroundColor White
                Write-Host "      3. 检查emotion2vec模型文件是否完整" -ForegroundColor White
                continue
            }

            # 步骤7: 清理临时文件
            Write-Host "    Step 7: Cleaning up temporary files..." -ForegroundColor Green
            Remove-Item $noisyAudioPath -Force -Recurse -ErrorAction SilentlyContinue

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

# 第二类：每个样本随机注入不同类型的噪声 (已注释)
<#
Write-Host "========================================" -ForegroundColor Blue
Write-Host "第二类：每个样本随机注入不同类型的噪声" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

foreach ($snr_db in $snr_levels) {
    $random_start_time = Get-Date
    $output_path = "$output_base_path\root2-${snr_db}db"
    
    Write-Host "Processing random noise at ${snr_db}dB..." -ForegroundColor Yellow
    Write-Host "Output: $output_path" -ForegroundColor Gray
    
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

        # 步骤1: 提取标签和生成初始Manifest
        Write-Host "  Step 1: Extracting emotion labels..." -ForegroundColor Green

        # 处理每个Session的情感标签
        for ($index = 1; $index -le 5; $index++) {
            $sessionPath = Join-Path $IEMOCAP_ROOT "Session$index\dialog\EmoEvaluation"
            $outputFile = Join-Path $output_path "Session$index.emo"
            
            if (Test-Path $sessionPath) {
                $txtFiles = Get-ChildItem -Path $sessionPath -Filter "*.txt"
                $allLines = @()
                
                foreach ($file in $txtFiles) {
                    $content = Get-Content $file.FullName
                    foreach ($line in $content) {
                        if ($line -match "Ses") {
                            $parts = $line -split "`t"
                            if ($parts.Length -ge 3) {
                                $col2 = $parts[1].Trim()
                                $col3 = $parts[2].Trim()
                                
                                if ($col3 -eq "ang" -or $col3 -eq "exc" -or $col3 -eq "hap" -or $col3 -eq "neu" -or $col3 -eq "sad") {
                                    if ($col3 -eq "exc") { 
                                        $col3 = "hap" 
                                    }
                                    $allLines += "$col2`t$col3"
                                }
                            }
                        }
                    }
                }
                
                $allLines | Out-File -FilePath $outputFile -Encoding UTF8
            }
        }

        # 合并所有Session文件到train.emo
        $trainFile = Join-Path $output_path "train.emo"
        $allTrainLines = @()

        for ($index = 1; $index -le 5; $index++) {
            $sessionFile = Join-Path $output_path "Session$index.emo"
            if (Test-Path $sessionFile) {
                $lines = Get-Content $sessionFile
                $allTrainLines += $lines
                Remove-Item $sessionFile
            }
        }

        $allTrainLines | Out-File -FilePath $trainFile -Encoding UTF8

        # 生成初始manifest文件
        $pythonScript = "scripts/iemocap_manifest.py"
        if (Test-Path $pythonScript) {
            python $pythonScript --root $IEMOCAP_ROOT --dest $output_path --label_path $trainFile
        } else {
            Write-Error "Python script not found: $pythonScript"
            continue
        }

        # 步骤2: 添加随机类型的真实噪声
        Write-Host "  Step 2: Adding random noise types (SNR=${snr_db}dB)..." -ForegroundColor Green

        $addNoiseScript = "scripts/add_real_noise_to_audio.py"
        if (Test-Path $addNoiseScript) {
            $arguments = @(
                $addNoiseScript,
                "--input_root", $IEMOCAP_ROOT,
                "--output_root", $noisyAudioPath,
                "--noise_root", $noise_root,
                "--snr_db", $snr_db,
                "--manifest_path", "$output_path\train.tsv",
                "--noise_mode", "random"
            )
            
            $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
            if ($process.ExitCode -ne 0) {
                Write-Error "Failed to add random noise. Exit code: $($process.ExitCode)"
                continue
            }
        } else {
            Write-Error "Noise addition script not found: $addNoiseScript"
            continue
        }

        # 步骤3: 重新生成Manifest文件
        Write-Host "  Step 3: Regenerating manifest for noisy audio..." -ForegroundColor Green

        $noisyManifestScript = "scripts/iemocap_manifest_noisy.py"
        if (Test-Path $noisyManifestScript) {
            python $noisyManifestScript --root $noisyAudioPath --dest $output_path --label_path $trainFile
        } else {
            Write-Error "Noisy manifest script not found: $noisyManifestScript"
            continue
        }

        # 步骤4: 检查和修复音频格式
        Write-Host "  Step 4: Checking and fixing audio format..." -ForegroundColor Green

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

        # 步骤5: 特征提取
        Write-Host "  Step 5: Extracting features from noisy audio..." -ForegroundColor Green

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
            $exit_code = $process.ExitCode
            Write-Error "Feature extraction failed with exit code: $exit_code"
            Write-Host "  尝试的解决方案:" -ForegroundColor Yellow
            Write-Host "    1. 检查音频文件是否为16kHz单声道格式" -ForegroundColor White
            Write-Host "    2. 检查CUDA环境是否正常" -ForegroundColor White
            Write-Host "    3. 检查emotion2vec模型文件是否完整" -ForegroundColor White
            continue
        }

        # 步骤6: 清理临时文件
        Write-Host "  Step 6: Cleaning up temporary files..." -ForegroundColor Green
        Remove-Item $noisyAudioPath -Force -Recurse -ErrorAction SilentlyContinue

        $random_end_time = Get-Date
        $random_duration = $random_end_time - $random_start_time
        $random_duration_minutes = $random_duration.TotalMinutes.ToString('0.1')
        Write-Host "  Completed: random noise at ${snr_db}dB in $random_duration_minutes minutes" -ForegroundColor Green
        
        $processed_count++
        
    } catch {
        $error_msg = $_.Exception.Message
        Write-Error "  Error processing random noise at ${snr_db}dB: $error_msg"
        $failed_count++
        continue
    }
}
#>

# 全局完成总结
$global_end_time = Get-Date
$global_duration = $global_end_time - $global_start_time

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Real Noise Preprocessing Pipeline Completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host ""
Write-Host "Processing Summary:" -ForegroundColor Cyan
# 只计算第一类噪声 (按类型分别处理)，第二类已注释
$total_expected = ($noise_types.Count * $snr_levels.Count)
Write-Host "  Total expected datasets: $total_expected (仅第一类: 按噪声类型分别处理)" -ForegroundColor White
Write-Host "  Successfully processed: $processed_count" -ForegroundColor Green
Write-Host "  Failed: $failed_count" -ForegroundColor Red
$global_duration_minutes = $global_duration.TotalMinutes.ToString('0.1')
Write-Host "  Total duration: $global_duration_minutes minutes" -ForegroundColor White

Write-Host ""
Write-Host "Real noise preprocessing pipeline completed!" -ForegroundColor Green 