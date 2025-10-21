param(
    [Parameter(Mandatory=$true)]
    [string]$CASIA_ROOT,
    
    [Parameter(Mandatory=$false)]
    [string]$output_base_path = "C:\Users\admin\Desktop\DATA",
    
    [Parameter(Mandatory=$false)]
    [string]$noise_dir = "C:\Users\admin\Desktop\DATA\noise-92\NOISE\data\signals\5types",
    
    [Parameter(Mandatory=$false)]
    [array]$noise_types = @("babble", "factory1", "hfchannel", "volvo"),
    
    [Parameter(Mandatory=$false)]
    [array]$snr_levels = @(0, 5, 10, 15, 20)
)

# 运行命令示例:
# .\real_noise_preprocessing.ps1 -CASIA_ROOT "C:\Users\admin\Desktop\DATA\CASIA\CASIA情感语料库" -output_base_path "C:\Users\admin\Desktop\DATA\processed_features_CASIA_real_noise" -noise_types @("babble", "factory1", "hfchannel", "volvo") -snr_levels @(20,15,10,0)

Write-Host "========================================" -ForegroundColor Green
Write-Host "Multi-Noise-Type CASIA Real Noise Audio Preprocessing Pipeline" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 检查输入参数
if (!(Test-Path $CASIA_ROOT)) {
    Write-Error "CASIA root directory not found: $CASIA_ROOT"
    exit 1
}

if (!(Test-Path $noise_dir)) {
    Write-Error "Noise directory not found: $noise_dir"
    exit 1
}

Write-Host "Input CASIA Root: $CASIA_ROOT" -ForegroundColor Cyan
Write-Host "Output Base Directory: $output_base_path" -ForegroundColor Cyan
Write-Host "Noise Directory: $noise_dir" -ForegroundColor Cyan
Write-Host "Noise Types: $($noise_types -join ', ')" -ForegroundColor Cyan
Write-Host "SNR Levels: $($snr_levels -join ', ') dB" -ForegroundColor Cyan
Write-Host ""

# 检查噪声文件是否存在
Write-Host "Verifying noise files..." -ForegroundColor Yellow
foreach ($noise_type in $noise_types) {
    $noise_file = Join-Path $noise_dir "${noise_type}.wav"
    if (!(Test-Path $noise_file)) {
        Write-Error "Noise file not found: $noise_file"
        exit 1
    }
    Write-Host "✓ Found: ${noise_type}.wav" -ForegroundColor Green
}
Write-Host ""

$global_start_time = Get-Date
$total_combinations = $noise_types.Count * $snr_levels.Count
$processed_count = 0
$failed_count = 0

Write-Host "Total processing combinations: $total_combinations (${($noise_types.Count)} noise types × ${($snr_levels.Count)} SNR levels)" -ForegroundColor Cyan
Write-Host ""

# 为每个噪声类型和SNR水平组合进行处理
foreach ($noise_type in $noise_types) {
    foreach ($snr_db in $snr_levels) {
        $combination_start_time = Get-Date
        $output_path = "$output_base_path\processed_features_${noise_type}_${snr_db}db"
        
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "Processing: $noise_type noise at $snr_db dB SNR" -ForegroundColor Yellow
        Write-Host "Output: $output_path" -ForegroundColor Yellow
        Write-Host "Progress: $($processed_count + $failed_count + 1)/$total_combinations" -ForegroundColor Yellow
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
            # 步骤1: 生成CASIA标签和初始Manifest
            # ==========================================
            Write-Host "Step 1: Generating CASIA labels and initial manifest..." -ForegroundColor Green

            # 生成初始manifest文件（基于原始音频）
            Write-Host "Generating initial manifest file..." -ForegroundColor White
            $pythonScript = "scripts/casia_manifest.py"
            if (Test-Path $pythonScript) {
                & python $pythonScript --root $CASIA_ROOT --dest $output_path
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
            # 步骤2: 添加真实噪声到音频文件
            # ==========================================
            Write-Host ""
            Write-Host "Step 2: Adding real noise ($noise_type) to audio files (SNR=$snr_db dB)..." -ForegroundColor Green

            if ($snr_db -eq 0) {
                Write-Host "SNR = 0 dB: Adding maximum noise (signal equals noise power)" -ForegroundColor Yellow
            }

            $addRealNoiseScript = "scripts/add_real_noise_to_audio.py"
            if (Test-Path $addRealNoiseScript) {
                Write-Host "Processing audio files with $noise_type noise..." -ForegroundColor White
                Write-Host "This may take a while depending on the number of audio files..." -ForegroundColor Yellow
                
                $cmd = "& python `"$addRealNoiseScript`" --input_root `"$CASIA_ROOT`" --output_root `"$noisyAudioPath`" --noise_dir `"$noise_dir`" --noise_type `"$noise_type`" --snr_db $snr_db --manifest_path `"$output_path\train.tsv`""
                
                $process = Invoke-Expression $cmd
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Successfully added $noise_type noise (SNR=$snr_db dB)!" -ForegroundColor Green
                } else {
                    Write-Error "Failed to add $noise_type noise to audio files. Exit code: $LASTEXITCODE"
                    continue
                }
            } else {
                Write-Error "Real noise addition script not found: $addRealNoiseScript"
                continue
            }

            # ==========================================
            # 步骤3: 重新生成Manifest文件（基于噪声音频）
            # ==========================================
            Write-Host ""
            Write-Host "Step 3: Regenerating manifest for noisy audio files..." -ForegroundColor Green

            $noisyManifestScript = "scripts/casia_manifest_noisy.py"
            if (Test-Path $noisyManifestScript) {
                & python $noisyManifestScript --root $noisyAudioPath --original-manifest-dir $output_path --dest $output_path
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

            # 检查必要文件
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

            Write-Host "Starting feature extraction from $noise_type noisy audio (Layer 11)..." -ForegroundColor White
            Write-Host "This may take a while depending on your GPU and dataset size..." -ForegroundColor Yellow

            $cmd = "& python `"scripts/emotion2vec_speech_features.py`" --data `"$output_path`" --model `"$modelPath`" --split=train --checkpoint=`"$checkpointPath`" --save-dir=`"$output_path`" --layer=11"

            $process = Invoke-Expression $cmd
            if ($LASTEXITCODE -eq 0) {
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
                Write-Error "Feature extraction failed with exit code: $LASTEXITCODE"
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

            # 组合处理完成
            $combination_end_time = Get-Date
            $combination_duration = $combination_end_time - $combination_start_time
            Write-Host ""
            Write-Host "✅ $noise_type noise at $snr_db dB processing completed!" -ForegroundColor Green
            Write-Host "⏱️  Duration: $($combination_duration.TotalMinutes.ToString('0.1')) minutes" -ForegroundColor Green
            Write-Host "📁 Output: $output_path" -ForegroundColor Green
            Write-Host ""
            
            $processed_count++
            
        } catch {
            Write-Error "❌ Error processing $noise_type noise at $snr_db dB: $_"
            $failed_count++
            continue
        }
    }
}

# ==========================================
# 全局完成总结
# ==========================================
$global_end_time = Get-Date
$global_duration = $global_end_time - $global_start_time

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Multi-Noise-Type CASIA Real Noise Preprocessing Pipeline Completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host ""
Write-Host "📊 Processing Summary:" -ForegroundColor Cyan
Write-Host "  Total combinations: $total_combinations" -ForegroundColor White
Write-Host "  Successfully processed: $processed_count" -ForegroundColor Green
Write-Host "  Failed: $failed_count" -ForegroundColor Red
Write-Host "  Total duration: $($global_duration.TotalMinutes.ToString('0.1')) minutes" -ForegroundColor White

Write-Host ""
Write-Host "📁 Generated datasets:" -ForegroundColor Cyan
foreach ($noise_type in $noise_types) {
    foreach ($snr_db in $snr_levels) {
        $dataset_path = "$output_base_path\processed_features_${noise_type}_${snr_db}db"
        if (Test-Path $dataset_path) {
            $dataFiles = Get-ChildItem $dataset_path -File | Measure-Object -Property Length -Sum
            $totalSizeMB = [math]::Round($dataFiles.Sum / 1MB, 1)
            Write-Host "  ✅ $noise_type @ $snr_db dB: $dataset_path ($totalSizeMB MB)" -ForegroundColor Green
        } else {
            Write-Host "  ❌ $noise_type @ $snr_db dB: Failed to generate" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Run inference script to test model robustness across different noise types and SNR levels" -ForegroundColor White
Write-Host "  2. Compare accuracy degradation patterns between different noise types" -ForegroundColor White
Write-Host "  3. Analyze which emotion classes are most affected by each noise type" -ForegroundColor White
Write-Host "  4. Create cross-domain robustness analysis comparing real vs synthetic noise" -ForegroundColor White

Write-Host ""
Write-Host "🎉 Multi-noise-type CASIA preprocessing pipeline completed!" -ForegroundColor Green
Write-Host "You can now test your model's robustness across different real-world noise conditions." -ForegroundColor Cyan

Write-Host ""
Write-Host "📈 Suggested Analysis:" -ForegroundColor Magenta
Write-Host "  • babble: Human speech interference (cocktail party effect)" -ForegroundColor White
Write-Host "  • factory1: Industrial/mechanical noise" -ForegroundColor White
Write-Host "  • hfchannel: Communication channel distortion" -ForegroundColor White
Write-Host "  • volvo: Automotive/road noise" -ForegroundColor White