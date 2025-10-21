param(
    [Parameter(Mandatory=$true)]
    [string]$IEMOCAP_ROOT,
    
    [Parameter(Mandatory=$true)]
    [string]$output_path
)
#运行命令-IEMOCAP_ROOT“your path” -output_path "your path"
#.\complete_preprocessing.ps1 -IEMOCAP_ROOT "C:\Users\admin\Desktop\DATA\IEMOCAP_full_release" -output_path "C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP"
Write-Host "========================================" -ForegroundColor Green
Write-Host "IEMOCAP Complete Preprocessing Pipeline" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 检查输入参数
if (!(Test-Path $IEMOCAP_ROOT)) {
    Write-Error "IEMOCAP root directory not found: $IEMOCAP_ROOT"
    exit 1
}

Write-Host "Input IEMOCAP Root: $IEMOCAP_ROOT" -ForegroundColor Cyan
Write-Host "Output Directory: $output_path" -ForegroundColor Cyan
Write-Host ""

# 创建输出目录
if (!(Test-Path $output_path)) {
    New-Item -ItemType Directory -Path $output_path -Force | Out-Null
    Write-Host "Created output directory: $output_path" -ForegroundColor Yellow
} else {
    Write-Host "Cleaning existing output directory..." -ForegroundColor Yellow
    Remove-Item "$output_path\*" -Force -ErrorAction SilentlyContinue
}

# ==========================================
# 步骤1: 标签提取和Manifest生成
# ==========================================
Write-Host ""
Write-Host "Step 1: Extracting emotion labels and generating manifest..." -ForegroundColor Green

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

# 调用Python脚本生成manifest
Write-Host "Generating manifest file..." -ForegroundColor White
$pythonScript = "scripts/iemocap_manifest.py"
if (Test-Path $pythonScript) {
    python $pythonScript --root $IEMOCAP_ROOT --dest $output_path --label_path $trainFile
    if (Test-Path "$output_path\train.tsv") {
        $tsvLines = (Get-Content "$output_path\train.tsv").Count - 1
        Write-Host "Generated train.tsv with $tsvLines audio files" -ForegroundColor Green
    } else {
        Write-Error "Failed to generate train.tsv"
        exit 1
    }
} else {
    Write-Error "Python script not found: $pythonScript"
    exit 1
}

# ==========================================
# 步骤2: 特征提取
# ==========================================
Write-Host ""
Write-Host "Step 2: Extracting features from emotion2vec model..." -ForegroundColor Green

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
    exit 1
}

# 使用现有的upstream目录，包含完整的任务定义
$modelPath = "upstream"
if (!(Test-Path $modelPath)) {
    Write-Error "Upstream directory not found: $modelPath"
    Write-Host "Please ensure the upstream directory with task definitions exists." -ForegroundColor Red
    exit 1
}
Write-Host "Using existing upstream directory with task definitions: $modelPath" -ForegroundColor Green

Write-Host "Starting feature extraction (Layer 12)..." -ForegroundColor White
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

try {
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
        exit $process.ExitCode
    }
} catch {
    Write-Error "Error during feature extraction: $_"
    exit 1
}

# ==========================================
# 清理临时文件
# ==========================================
Write-Host ""
Write-Host "Cleaning up temporary files..." -ForegroundColor Green
# 不需要删除upstream目录，因为它是项目的一部分
Write-Host "No temporary files to clean up" -ForegroundColor Yellow

# ==========================================
# 完成总结
# ==========================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Preprocessing Pipeline Completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host ""
Write-Host "Generated files in $output_path :" -ForegroundColor Cyan
$outputFiles = Get-ChildItem $output_path | Sort-Object Name
foreach ($file in $outputFiles) {
    $size = if ($file.PSIsContainer) { "Directory" } else { 
        [math]::Round($file.Length / 1MB, 2).ToString() + " MB" 
    }
    Write-Host "  $($file.Name) - $size" -ForegroundColor White
}

Write-Host ""
Write-Host "Preprocessing completed successfully!" -ForegroundColor Green
Write-Host "You can now use these files for training your emotion recognition model." -ForegroundColor Cyan 