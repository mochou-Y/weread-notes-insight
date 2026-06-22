#!/bin/bash

# 微信读书笔记聚类+图谱生成脚本
# 用法: ./run_cluster.sh [min-cluster-size] [min-samples] [method] [n-components]
# 示例: ./run_cluster.sh 3 2 eom 15
#       ./run_cluster.sh 5 3 leaf 20
#       ./run_cluster.sh

# 设置编码为UTF-8
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

# 默认参数
MIN_CLUSTER_SIZE=${1:-3}
MIN_SAMPLES=${2:-2}
METHOD=${3:-eom}
N_COMPONENTS=${4:-15}

# 创建log文件夹
mkdir -p log

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log/cluster_${TIMESTAMP}.log"

# 获取output文件夹中test_graph_xxx.html的最大序号
get_next_index() {
    local max_index=-1
    if [ -d "output" ]; then
        for file in output/test_graph_*.html; do
            if [ -f "$file" ]; then
                # 提取文件名中的数字部分
                filename=$(basename "$file")
                index=$(echo "$filename" | sed 's/test_graph_\([0-9]*\)\.html/\1/')
                # 移除前导零
                index=$((10#$index))
                if [ "$index" -gt "$max_index" ]; then
                    max_index=$index
                fi
            fi
        done
    fi
    # 下一个序号，格式化为3位数字
    printf "%03d" $((max_index + 1))
}

# 获取下一个图谱文件序号
NEXT_INDEX=$(get_next_index)
OUTPUT_FILE="test_graph_${NEXT_INDEX}.html"

echo "========================================" | tee -a "$LOG_FILE"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "参数: min_cluster_size=$MIN_CLUSTER_SIZE, min_samples=$MIN_SAMPLES, method=$METHOD, n_components=$N_COMPONENTS" | tee -a "$LOG_FILE"
echo "图谱输出: $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 执行聚类命令
echo "" | tee -a "$LOG_FILE"
echo "[命令1] 执行聚类..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
START_TIME=$(date +%s)
"$PYTHON_BIN" -m src.main cluster \
    --min-cluster-size "$MIN_CLUSTER_SIZE" \
    --min-samples "$MIN_SAMPLES" \
    --method "$METHOD" \
    --n-components "$N_COMPONENTS" 2>&1 | tee -a "$LOG_FILE"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))
echo "" | tee -a "$LOG_FILE"
if [ $MINUTES -gt 0 ]; then
    echo "聚类耗时: ${MINUTES}分钟${SECONDS}秒" | tee -a "$LOG_FILE"
else
    echo "聚类耗时: ${SECONDS}秒" | tee -a "$LOG_FILE"
fi

# 执行图谱生成命令
START_TIME_PHASE_2=$(date +%s)
echo "" | tee -a "$LOG_FILE"
echo "[命令2] 生成图谱..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
"$PYTHON_BIN" -m src.main graph --output "$OUTPUT_FILE" 2>&1 | tee -a "$LOG_FILE"

END_TIME_PHASE_2=$(date +%s)
ELAPSED_PHASE_2=$((END_TIME_PHASE_2 - START_TIME_PHASE_2))
MINUTES_PHASE_2=$((ELAPSED_PHASE_2 / 60))
SECONDS_PHASE_2=$((ELAPSED_PHASE_2 % 60))
echo "" | tee -a "$LOG_FILE"
if [ $MINUTES_PHASE_2 -gt 0 ]; then
    echo "生成图谱耗时: ${MINUTES_PHASE_2}分钟${SECONDS_PHASE_2}秒" | tee -a "$LOG_FILE"
else
    echo "生成图谱耗时: ${SECONDS_PHASE_2}秒" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "完成! 日志已保存到: $LOG_FILE" | tee -a "$LOG_FILE"
echo "图谱已保存到: output/$OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
