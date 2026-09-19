param(
    [int[]]$Shots = @(4, 8, 16),
    [int[]]$Seeds = @(42, 43, 44, 45),
    [string]$Python = "C:\Users\Administrator\miniconda3\envs\LLM\python.exe"
)

$ErrorActionPreference = "Stop"
$project = "E:\nlp_arch_lab"
$benchmark = Join-Path $project "src\benchmark.py"

foreach ($shot in $Shots) {
    $out = Join-Path $project ("runs\reproduce_global_{0}shot_{1}seed" -f $shot, $Seeds.Count)
    & $Python $benchmark --dataset ag_news_local --shots $shot --epochs 1 --seeds $Seeds --max-len 128 --tokenizer word --evidence-routing global --sifter-residual-scale 1.0 --output-dir $out
}

# Secondary real-data task. This is separate from the AG News headline table:
# SST-2 hashword2 was introduced after the word-tokenizer exploratory result.
$sstOut = Join-Path $project ("runs\reproduce_sst2_hashword2_8shot_{0}seed" -f $Seeds.Count)
& $Python $benchmark --dataset sst2_local --shots 8 --epochs 1 --seeds $Seeds --max-len 96 --tokenizer hashword2 --evidence-routing global --sifter-residual-scale 1.0 --output-dir $sstOut

# Third real-data task. TREC uses the word tokenizer and both 4/8-shot points.
foreach ($trecShot in @(4, 8)) {
    $trecOut = Join-Path $project ("runs\reproduce_trec_word_{0}shot_{1}seed" -f $trecShot, $Seeds.Count)
    & $Python $benchmark --dataset trec_local --shots $trecShot --epochs 1 --seeds $Seeds --max-len 96 --tokenizer word --evidence-routing global --sifter-residual-scale 1.0 --output-dir $trecOut
}

# Optional controlled long-range diagnostic. It is separate from the real-data headline table.
$challengeOut = Join-Path $project ("runs\reproduce_challenge_positional_8shot_{0}seed" -f $Seeds.Count)
& $Python $benchmark --dataset challenge --shots 8 --epochs 1 --seeds $Seeds --max-len 128 --tokenizer word --evidence-routing positional --output-dir $challengeOut
