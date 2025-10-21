param(
    [Parameter(Mandatory=$true)]
    [string]$emotion2vec_path,
    
    [Parameter(Mandatory=$true)]
    [string]$manifest_path,
    
    [Parameter(Mandatory=$true)]
    [string]$model_path,
    
    [Parameter(Mandatory=$true)]
    [string]$checkpoint_path,
    
    [Parameter(Mandatory=$true)]
    [string]$save_dir
)

# 设置环境变量
$env:CUDA_LAUNCH_BLOCKING = "1"
$env:HYDRA_FULL_ERROR = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

# 设置PYTHONPATH
$originalPythonPath = $env:PYTHONPATH
if ($originalPythonPath) {
    $env:PYTHONPATH = "$emotion2vec_path;$originalPythonPath"
} else {
    $env:PYTHONPATH = $emotion2vec_path
}

Write-Host "Environment variables set:"
Write-Host "CUDA_LAUNCH_BLOCKING: $env:CUDA_LAUNCH_BLOCKING"
Write-Host "HYDRA_FULL_ERROR: $env:HYDRA_FULL_ERROR"
Write-Host "CUDA_VISIBLE_DEVICES: $env:CUDA_VISIBLE_DEVICES"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
Write-Host ""

# 检查必要文件是否存在
if (!(Test-Path $manifest_path)) {
    Write-Error "Manifest file not found: $manifest_path"
    exit 1
}

if (!(Test-Path $model_path)) {
    Write-Error "Model file not found: $model_path"
    exit 1
}

if (!(Test-Path $checkpoint_path)) {
    Write-Error "Checkpoint file not found: $checkpoint_path"
    exit 1
}

# 创建保存目录
if (!(Test-Path $save_dir)) {
    New-Item -ItemType Directory -Path $save_dir -Force
    Write-Host "Created save directory: $save_dir"
}

# 检查Python脚本是否存在
$pythonScript = "scripts/emotion2vec_speech_features.py"
if (!(Test-Path $pythonScript)) {
    Write-Error "Python script not found: $pythonScript"
    exit 1
}

Write-Host "Starting feature extraction..."
Write-Host "Manifest path: $manifest_path"
Write-Host "Model path: $model_path"
Write-Host "Checkpoint path: $checkpoint_path"
Write-Host "Save directory: $save_dir"
Write-Host ""

# 只提取最后一层 (layer 11)
for ($layer = 11; $layer -le 11; $layer++) {
    $true_layer = $layer + 1
    Write-Host "Extracting features from layer $true_layer"
    
    $arguments = @(
        $pythonScript,
        "--data", $manifest_path,
        "--model", $model_path,
        "--split=train",
        "--checkpoint=$checkpoint_path",
        "--save-dir=$save_dir",
        "--layer=$layer"
    )
    
    Write-Host "Running command: python $($arguments -join ' ')"
    
    try {
        $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow
        if ($process.ExitCode -eq 0) {
            Write-Host "Successfully extracted features from layer $true_layer" -ForegroundColor Green
        } else {
            Write-Error "Feature extraction failed with exit code: $($process.ExitCode)"
            exit $process.ExitCode
        }
    } catch {
        Write-Error "Error running Python script: $_"
        exit 1
    }
}

# 恢复原始PYTHONPATH
$env:PYTHONPATH = $originalPythonPath

Write-Host "Feature extraction completed successfully!" -ForegroundColor Green 