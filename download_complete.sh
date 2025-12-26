#!/bin/bash
# 功能：自动校验并下载完整的 BattlefieldKitDesert.z03 文件
# 适用：解决 huggingface-cli 下载文件大小不匹配、镜像源残缺问题

# ====================== 配置项（可根据需求修改）======================
# 数据集仓库信息
REPO_ID="wangxiangyu0814/TravelUAV"
REPO_TYPE="dataset"
# 目标文件名（如需下载其他分卷，修改此处即可，如 z01/z02）
TARGET_FILE="BattlefieldKitDesert.z03"
# 本地保存目录
LOCAL_DIR="/home/yyx/TravelUAV/data/TravelUAV_data"
# 文件预期大小（单位：字节，4.04G ≈ 4344406016 字节，可根据实际调整）
EXPECT_SIZE=4344406016
# 下载超时时间（秒）
TIMEOUT=1200
# ====================================================================

# 激活 conda 环境（确保 huggingface-cli 可用）
echo "【1/6】激活 llamauav 环境..."
source /home/yyx/miniconda3/etc/profile.d/conda.sh
conda activate llamauav || {
    echo "ERROR: 无法激活 llamauav 环境，请检查环境名是否正确！"
    exit 1
}

# 进入本地目录，检查文件是否存在
echo -e "\n【2/6】检查本地文件状态..."
cd $LOCAL_DIR || {
    echo "ERROR: 本地目录 $LOCAL_DIR 不存在！"
    exit 1
}

# 获取本地文件实际大小（字节）
if [ -f "$TARGET_FILE" ]; then
    ACTUAL_SIZE=$(ls -l "$TARGET_FILE" | awk '{print $5}')
    echo "本地文件 $TARGET_FILE 大小：$ACTUAL_SIZE 字节"
    echo "预期文件大小：$EXPECT_SIZE 字节"
    
    # 对比大小，判断是否完整
    if [ "$ACTUAL_SIZE" -eq "$EXPECT_SIZE" ]; then
        echo -e "\n✅ 文件已完整下载，无需重新下载！"
        exit 0
    else
        echo -e "\n❌ 文件残缺（大小不匹配），准备删除并重新下载..."
        rm -f "$TARGET_FILE" || {
            echo "ERROR: 无法删除残缺文件，请检查权限！"
            exit 1
        }
    fi
else
    echo -e "\n⚠️ 本地未找到 $TARGET_FILE，准备直接下载..."
fi

# 清除 huggingface 缓存（避免缓存干扰）
echo -e "\n【3/6】清除 huggingface 缓存..."
huggingface-cli cache clear > /dev/null 2>&1
huggingface-cli cache delete $REPO_ID --repo-type $REPO_TYPE > /dev/null 2>&1

# 切换到官方源，强制重新下载
echo -e "\n【4/6】切换 Hugging Face 官方源，开始下载..."
unset HF_ENDPOINT
huggingface-cli download \
    $REPO_ID \
    --repo-type $REPO_TYPE \
    --local-dir $LOCAL_DIR \
    --include "$TARGET_FILE" \
    --force-download \
    --timeout $TIMEOUT || {
        echo -e "\nERROR: 文件下载失败，请检查网络或仓库权限！"
        exit 1
    }

# 再次校验文件大小
echo -e "\n【5/6】校验下载后的文件完整性..."
if [ -f "$TARGET_FILE" ]; then
    FINAL_SIZE=$(ls -l "$TARGET_FILE" | awk '{print $5}')
    if [ "$FINAL_SIZE" -eq "$EXPECT_SIZE" ]; then
        echo -e "\n✅ 最终文件大小：$FINAL_SIZE 字节（与预期一致）"
        echo -e "\n🎉 文件 $TARGET_FILE 已完整下载！"
    else
        echo -e "\n❌ 下载后文件仍残缺！"
        echo "最终大小：$FINAL_SIZE 字节，预期：$EXPECT_SIZE 字节"
        echo "建议：使用 wget 断点续传下载（脚本末尾有示例）"
        exit 1
    fi
else
    echo -e "\n❌ 下载后未找到文件 $TARGET_FILE！"
    exit 1
fi

# 输出文件信息
echo -e "\n【6/6】下载完成，文件信息："
ls -lh "$TARGET_FILE"

# 可选：wget 断点续传兜底方案（如需手动执行，取消注释）
# echo -e "\n📌 若仍下载失败，执行以下 wget 命令："
# echo "wget -c \"https://huggingface.co/datasets/$REPO_ID/resolve/main/$TARGET_FILE\" -P $LOCAL_DIR"

exit 0