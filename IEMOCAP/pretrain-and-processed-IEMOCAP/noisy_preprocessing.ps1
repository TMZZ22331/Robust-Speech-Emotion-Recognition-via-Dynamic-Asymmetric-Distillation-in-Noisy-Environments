param(
    [Parameter(Mandatory=$true)]
    [string]$IEMOCAP_ROOT,
    
    [Parameter(Mandatory=$false)]
    [string]$output_base_path = "C:\Users\admin\Desktop\DATA",
    
    [Parameter(Mandatory=$false)]
    [array]$snr_levels = @(0, 5, 10, 15, 20)
)
#运行命令-IEMOCAP_ROOT“your path” -output_base_path "your path" -snr_levels @(20)
#.\noisy_preprocessing.ps1 -IEMOCAP_ROOT "C:\Users\admin\Desktop\DATA\IEMOCAP_full_release" -output_base_path "C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy" -snr_levels @(20,15,10,0)
Write-Host "========================================" -ForegroundColor Green
Write-Host "Multi-SNR IEMOCAP Noisy Audio Preprocessing Pipeline" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 检查输入参数
if (!(Test-Path $IEMOCAP_ROOT)) {
    Write-Error "IEMOCAP root directory not found: $IEMOCAP_ROOT"
    exit 1
}

Write-Host "Input IEMOCAP Root: $IEMOCAP_ROOT" -ForegroundColor Cyan
Write-Host "Output Base Directory: $output_base_path" -ForegroundColor Cyan
Write-Host "SNR Levels: $($snr_levels -join ', ') dB" -ForegroundColor Cyan
Write-Host ""

$global_start_time = Get-Date
$processed_snr_count = 0
$failed_snr_count = 0

# 为每个SNR水平进行处理
foreach ($snr_db in $snr_levels) {
    $snr_start_time = Get-Date
    $output_path = "$output_base_path\processed_features_noisy_${snr_db}db"
    
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "Processing SNR: $snr_db dB" -ForegroundColor Yellow
    Write-Host "Output: $output_path" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        # 创建输出目录
        if (!(Test-Path $output_path)) {
            New-Item -ItemType Directory -Path $output_path -Force | Out-Null
            Write-Host "Created output directory: $output_path" -ForegroundColor Yellow
        } else {
            Write-Host "Cleaning existing output directory..." -ForegroundColor Yellow
            Remove-Item "$output_path\*" -Force -Recurse -ErrorAction SilentlyContinue
        }

        # 创建临时目录用于存储噪声音频
        $noisyAudioPath = "$output_path\noisy_audio_temp"
        New-Item -ItemType Directory -Path $noisyAudioPath -Force | Out-Null

        # ==========================================
        # 步骤1: 标签提取和初始Manifest生成
        # ==========================================
        Write-Host "Step 1: Extracting emotion labels and generating initial manifest..." -ForegroundColor Green

        # 处理每个Session
        Write-Host "Processing emotion labels from 5 sessions..." -ForegroundColor White
        for ($index = 1; $index -le 5; $index++) {
            $sessionPath = Join-Path $IEMOCAP_ROOT "Session$index\dialog\EmoEvaluation"
            $outputFile = Join-Path $output_path "Session$index.emo"
            
            Write-Host "  Processing Session$index..." -ForegroundColor Gray
            
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
                                    if ($col3 -eq "exc") { $col3 = "hap" }
                                    $allLines += "$col2`t$col3"
                                }
                            }
                        }
                    }
                }
                
                $allLines | Out-File -FilePath $outputFile -Encoding UTF8
                Write-Host "    Processed $($allLines.Count) emotion labels" -ForegroundColor Gray
            } else {
                Write-Warning "    Session path not found: $sessionPath"
            }
        }

        # 合并所有Session文件到train.emo
        Write-Host "Merging all sessions into train.emo..." -ForegroundColor White
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
        Write-Host "Created train.emo with $($allTrainLines.Count) total emotion labels" -ForegroundColor Green

        # 生成初始manifest文件（基于原始音频）
        Write-Host "Generating initial manifest file..." -ForegroundColor White
        $pythonScript = "scripts/iemocap_manifest.py"
        if (Test-Path $pythonScript) {
            python $pythonScript --root $IEMOCAP_ROOT --dest $output_path --label_path $trainFile
            if (Test-Path "$output_path\train.tsv") {
                $tsvLines = (Get-Content "$output_path\train.tsv").Count - 1
                Write-Host "Generated initial train.tsv with $tsvLines audio files" -ForegroundColor Green
            } else {
                Write-Error "Failed to generate initial train.tsv"
                continue
            }
        } else {
            Write-Error "Python script not found: $pythonScript"
            continue
        }

        # ==========================================
        # 步骤2: 添加白噪声到音频文件
        # ==========================================
        Write-Host ""
        Write-Host "Step 2: Adding white noise to audio files (SNR=$snr_db dB)..." -ForegroundColor Green

        if ($snr_db -eq 0) {
            Write-Host "SNR = 0 dB: Adding maximum noise (signal equals noise power)" -ForegroundColor Yellow
        }

        $addNoiseScript = "scripts/add_noise_to_audio.py"
        if (Test-Path $addNoiseScript) {
            Write-Host "Processing audio files with white noise..." -ForegroundColor White
            Write-Host "This may take a while depending on the number of audio files..." -ForegroundColor Yellow
            
            $arguments = @(
                $addNoiseScript,
                "--input_root", $IEMOCAP_ROOT,
                "--output_root", $noisyAudioPath,
                "--snr_db", $snr_db,
                "--manifest_path", "$output_path\train.tsv"
            )
            
            $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
            if ($process.ExitCode -eq 0) {
                Write-Host "Successfully added white noise (SNR=$snr_db dB)!" -ForegroundColor Green
            } else {
                Write-Error "Failed to add noise to audio files. Exit code: $($process.ExitCode)"
                continue
            }
        } else {
            Write-Error "Noise addition script not found: $addNoiseScript"
            continue
        }

        # ==========================================
        # 步骤3: 重新生成Manifest文件（基于噪声音频）
        # ==========================================
        Write-Host ""
        Write-Host "Step 3: Regenerating manifest for noisy audio files..." -ForegroundColor Green

        $noisyManifestScript = "scripts/iemocap_manifest_noisy.py"
        if (Test-Path $noisyManifestScript) {
            python $noisyManifestScript --root $noisyAudioPath --dest $output_path --label_path $trainFile
            if (Test-Path "$output_path\train.tsv") {
                $tsvLines = (Get-Content "$output_path\train.tsv").Count - 1
                Write-Host "Regenerated train.tsv with $tsvLines noisy audio files" -ForegroundColor Green
            } else {
                Write-Error "Failed to regenerate train.tsv for noisy audio"
                continue
            }
        } else {
            Write-Error "Noisy manifest script not found: $noisyManifestScript"
            continue
        }

        # ==========================================
        # 步骤4: 从噪声音频提取特征
        # ==========================================
        Write-Host ""
        Write-Host "Step 4: Extracting features from noisy audio using emotion2vec..." -ForegroundColor Green

        # 设置环境变量
        $env:CUDA_LAUNCH_BLOCKING = "1"
        $env:HYDRA_FULL_ERROR = "1"
        $env:CUDA_VISIBLE_DEVICES = "0"
        $env:PYTHONPATH = ".."

        Write-Host "Environment variables configured for GPU processing" -ForegroundColor White

        # 检查必要文件 - 修正路径
        $checkpointPath = "emotion2vec_base.pt"

        if (!(Test-Path $checkpointPath)) {
            Write-Error "Checkpoint file not found: $checkpointPath"
            continue
        }

        # 使用现有的upstream目录，包含完整的任务定义
        $modelPath = "upstream"
        if (!(Test-Path $modelPath)) {
            Write-Error "Upstream directory not found: $modelPath"
            Write-Host "Please ensure the upstream directory with task definitions exists." -ForegroundColor Red
            continue
        }
        Write-Host "Using existing upstream directory with task definitions: $modelPath" -ForegroundColor Green

        Write-Host "Starting feature extraction from noisy audio (Layer 12)..." -ForegroundColor White
        Write-Host "This may take a while depending on your GPU and dataset size..." -ForegroundColor Yellow

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
        if ($process.ExitCode -eq 0) {
            Write-Host "Feature extraction completed successfully!" -ForegroundColor Green
            
            # 检查输出文件
            $npyFile = Join-Path $output_path "train.npy"
            $lengthsFile = Join-Path $output_path "train.lengths"
            
            if (Test-Path $npyFile) {
                $npySize = [math]::Round((Get-Item $npyFile).Length / 1MB, 2)
                Write-Host "Generated features file: train.npy ($npySize MB)" -ForegroundColor Green
            }
            
            if (Test-Path $lengthsFile) {
                $lengthsCount = (Get-Content $lengthsFile).Count
                Write-Host "Generated lengths file: train.lengths ($lengthsCount entries)" -ForegroundColor Green
            }
        } else {
            Write-Error "Feature extraction failed with exit code: $($process.ExitCode)"
            continue
        }

        # ==========================================
        # 步骤5: 清理临时文件
        # ==========================================
        Write-Host ""
        Write-Host "Step 5: Cleaning up temporary files..." -ForegroundColor Green

        Write-Host "Removing temporary noisy audio files..." -ForegroundColor White
        Remove-Item $noisyAudioPath -Force -Recurse -ErrorAction SilentlyContinue

        Write-Host "Temporary files cleaned up" -ForegroundColor Green

        # SNR处理完成
        $snr_end_time = Get-Date
        $snr_duration = $snr_end_time - $snr_start_time
        Write-Host ""
        Write-Host "✅ SNR $snr_db dB processing completed!" -ForegroundColor Green
        Write-Host "⏱️  Duration: $($snr_duration.TotalMinutes.ToString('0.1')) minutes" -ForegroundColor Green
        Write-Host "📁 Output: $output_path" -ForegroundColor Green
        Write-Host ""
        
        $processed_snr_count++
        
    } catch {
        Write-Error "❌ Error processing SNR $snr_db dB: $_"
        $failed_snr_count++
        continue
    }
}

# ==========================================
# 全局完成总结
# ==========================================
$global_end_time = Get-Date
$global_duration = $global_end_time - $global_start_time

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Multi-SNR Preprocessing Pipeline Completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host ""
Write-Host "📊 Processing Summary:" -ForegroundColor Cyan
Write-Host "  Total SNR levels: $($snr_levels.Count)" -ForegroundColor White
Write-Host "  Successfully processed: $processed_snr_count" -ForegroundColor Green
Write-Host "  Failed: $failed_snr_count" -ForegroundColor Red
Write-Host "  Total duration: $($global_duration.TotalMinutes.ToString('0.1')) minutes" -ForegroundColor White

Write-Host ""
Write-Host "📁 Generated datasets:" -ForegroundColor Cyan
foreach ($snr_db in $snr_levels) {
    $dataset_path = "$output_base_path\processed_features_noisy_${snr_db}db"
    if (Test-Path $dataset_path) {
        $dataFiles = Get-ChildItem $dataset_path -File | Measure-Object -Property Length -Sum
        $totalSizeMB = [math]::Round($dataFiles.Sum / 1MB, 1)
        Write-Host "  ✅ SNR $snr_db dB: $dataset_path ($totalSizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ SNR $snr_db dB: Failed to generate" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Run inference script to test model robustness across SNR levels" -ForegroundColor White
Write-Host "  2. Compare accuracy degradation patterns" -ForegroundColor White
Write-Host "  3. Analyze which emotion classes are most affected by noise" -ForegroundColor White

Write-Host ""
Write-Host "🎉 Multi-SNR preprocessing pipeline completed!" -ForegroundColor Green
Write-Host "You can now test your model's robustness across different noise levels." -ForegroundColor Cyan 