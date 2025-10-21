param(
    [Parameter(Mandatory=$true)]
    [string]$IEMOCAP_ROOT,
    
    [Parameter(Mandatory=$true)]
    [string]$output_path
)

# 创建输出目录
if (!(Test-Path $output_path)) {
    New-Item -ItemType Directory -Path $output_path -Force
}

# 处理每个Session
for ($index = 1; $index -le 5; $index++) {
    $sessionPath = Join-Path $IEMOCAP_ROOT "Session$index\dialog\EmoEvaluation"
    $outputFile = Join-Path $output_path "Session$index.emo"
    
    Write-Host "Processing Session$index..."
    
    # 检查路径是否存在
    if (Test-Path $sessionPath) {
        # 获取所有.txt文件
        $txtFiles = Get-ChildItem -Path $sessionPath -Filter "*.txt"
        
        # 处理每个txt文件
        $allLines = @()
        foreach ($file in $txtFiles) {
            $content = Get-Content $file.FullName
            foreach ($line in $content) {
                # 检查是否包含"Ses"
                if ($line -match "Ses") {
                    # 分割行，获取第2和第3列（制表符分隔）
                    $parts = $line -split "`t"
                    if ($parts.Length -ge 3) {
                        $col2 = $parts[1].Trim()
                        $col3 = $parts[2].Trim()
                        
                        # 过滤情绪类型
                        if ($col3 -eq "ang" -or $col3 -eq "exc" -or $col3 -eq "hap" -or $col3 -eq "neu" -or $col3 -eq "sad") {
                            # 将exc替换为hap
                            if ($col3 -eq "exc") {
                                $col3 = "hap"
                            }
                            $allLines += "$col2`t$col3"
                        }
                    }
                }
            }
        }
        
        # 写入输出文件
        $allLines | Out-File -FilePath $outputFile -Encoding UTF8
        Write-Host "Processed $($allLines.Count) lines for Session$index"
    } else {
        Write-Warning "Path not found: $sessionPath"
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
        # 删除临时文件
        Remove-Item $sessionFile
    }
}

# 写入合并后的训练文件
$allTrainLines | Out-File -FilePath $trainFile -Encoding UTF8
Write-Host "Created train.emo with $($allTrainLines.Count) total lines"

# 调用Python脚本
Write-Host "Calling Python script..."
$pythonScript = "scripts/iemocap_manifest.py"
if (Test-Path $pythonScript) {
    python $pythonScript --root $IEMOCAP_ROOT --dest $output_path --label_path $trainFile
} else {
    Write-Warning "Python script not found: $pythonScript"
}

Write-Host "Processing completed!" 