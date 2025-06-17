#!/bin/bash

# ==============================================================================
# 全功能实验执行脚本
#
# 该脚本可以执行单个或全部实验变种。
#
# ---
#
# ## 使用方法:
#
# 1. **运行所有实验的所有变种 (使用下面定义的默认配置):**
#    bash run_all_experiments.sh
#
# 2. **运行指定的一个实验的所有变种 (例如 cnn):**
#    bash run_all_experiments.sh cnn
#
# 3. **运行指定的一个变种 (例如 cnn_resnet18_adam):**
#    bash run_all_experiments.sh cnn_resnet18_adam
#
# 4. **运行指定变种并覆盖参数:**
#    bash run_all_experiments.sh <variant_name> --epochs 50
#
#    示例: 运行 cnn_resnet18_adam 变种, 但训练50轮
#    bash run_all_experiments.sh cnn_resnet18_adam --epochs 50
#
# ---
#
# ==============================================================================

set -e # 如果任何命令失败，则立即退出
set -o pipefail # 确保管道中的命令失败时，脚本也会失败

# 激活 Conda 环境
source /root/data-tmp/miniconda3/etc/profile.d/conda.sh
conda activate llm

# ==============================================================================
# 1. 准备数据集
# ==============================================================================
# 在此预先下载所有数据集，以避免多进程训练时发生下载冲突。
echo "正在准备所有必需的数据集..."
python -c "from torchvision.datasets import MNIST, CIFAR10; print('正在下载 MNIST...'); MNIST(root='data/mnist', download=True, train=True); MNIST(root='data/mnist', download=True, train=False); print('正在下载 CIFAR-10...'); CIFAR10(root='data/cifar10', download=True, train=True); CIFAR10(root='data/cifar10', download=True, train=False); print('所有数据集准备完毕。')"
echo "------------------------------------------------------------------------------"

# 安装中文字体以支持 matplotlib 绘图
echo "正在检查并安装中文字体..."
if ! fc-list | grep -qi "WenQuanYi Zen Hei"; then
    apt-get update -qq && apt-get install -y -qq fonts-wqy-zenhei
    # 清除 matplotlib 字体缓存以确保新字体被加载
    rm -rf /root/.cache/matplotlib
    echo "字体安装和缓存清理完成。"
else
    echo "中文字体已安装。"
fi



# 自动检测并设置可用的GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 命令未找到。请确保 NVIDIA 驱动已正确安装。"
    exit 1
fi
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "警告: 未检测到GPU。将使用CPU进行训练。"
    NPROC_PER_NODE=1 # 对于CPU，使用单进程
else
    echo "检测到 $GPU_COUNT 个GPU。"
    NPROC_PER_NODE=$GPU_COUNT
fi
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))

# 创建日志目录
mkdir -p logs

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ==============================================================================
#   实验定义
# ==============================================================================
# 在关联数组中定义所有实验变种
# 格式: declare -A-g variant_<variant_name>=(
#          [script]="<path_to_script>"
#          [args]="<arguments>"
#       )
#
# -g 使得函数内部也可以访问这些定义
# ==============================================================================

# --- MLP 实验 (共4个变种) ---
declare -A -g variant_mlp_baseline=(
    [module]="src.mlp.train"
    [args]="--model_name baseline --optimizer sgd --lr 0.01 --epochs 5 --batch_size 64 --exp_name baseline"
)
declare -A -g variant_mlp_wider_net=(
    [module]="src.mlp.train"
    [args]="--model_name wider_net --optimizer sgd --lr 0.01 --epochs 5 --batch_size 64 --exp_name wider_net"
)
declare -A -g variant_mlp_deeper_net=(
    [module]="src.mlp.train"
    [args]="--model_name deeper_net --optimizer sgd --lr 0.01 --epochs 5 --batch_size 64 --exp_name deeper_net"
)
declare -A -g variant_mlp_adam_optimizer=(
    [module]="src.mlp.train"
    [args]="--model_name baseline --optimizer adam --lr 0.001 --epochs 5 --batch_size 64 --exp_name adam_optimizer"
)

# --- CNN 实验 (共4个变种) ---
declare -A -g variant_cnn_baseline=(
    [module]="src.cnn.train"
    [args]="--model_name baseline --optimizer sgd --lr 0.001 --epochs 10 --batch_size 128 --exp_name baseline"
)
declare -A -g variant_cnn_baseline_adam=(
    [module]="src.cnn.train"
    [args]="--model_name baseline --optimizer adam --lr 0.001 --epochs 10 --batch_size 128 --exp_name baseline_adam"
)
declare -A -g variant_cnn_baseline_sgd_high_lr=(
    [module]="src.cnn.train"
    [args]="--model_name baseline --optimizer sgd --lr 0.01 --epochs 10 --batch_size 128 --exp_name baseline_sgd_high_lr"
)
declare -A -g variant_cnn_resnet18_adam=(
    [module]="src.cnn.train"
    [args]="--model_name resnet18 --optimizer adam --lr 0.001 --epochs 10 --batch_size 256 --exp_name resnet18_adam"
)
declare -A -g variant_cnn_densenet121_adam=(
    [module]="src.cnn.train"
    [args]="--model_name densenet121 --optimizer adam --lr 0.001 --epochs 10 --batch_size 128 --exp_name densenet121_adam"
)
declare -A -g variant_cnn_se_resnet18_adam=(
    [module]="src.cnn.train"
    [args]="--model_name se_resnet18 --optimizer adam --lr 0.001 --epochs 10 --batch_size 256 --exp_name se_resnet18_adam"
)

# --- RNN 实验 (共2个变种) ---
declare -A -g variant_rnn_manual_surname=(
    [module]="src.rnn.train"
    [args]="--model rnn --exp_name surname_manual_rnn --n_iters 10000 --print_every 500"
)
declare -A -g variant_rnn_lstm_surname=(
    [module]="src.rnn.train"
    [args]="--model lstm --exp_name surname_lstm --n_iters 10000 --print_every 500"
)

# --- GAN 实验 (共2个变种) ---
declare -A -g variant_gan_dcgan=(
    [module]="src.gan.train"
    [args]="--epochs 25 --batch_size 256 --exp_name dcgan"
)
declare -A -g variant_gan_dcgan_mnist=(
    [module]="src.gan.train"
    [args]="--exp_name gan_dcgan_mnist --dataset mnist --epochs 15 --batch_size 64"
)
declare -A -g variant_gan_dcgan_fashion_mnist=(
    [module]="src.gan.train"
    [args]="--exp_name gan_dcgan_fashion_mnist --dataset fashion_mnist --epochs 15 --batch_size 64"
)

# --- MLP-Mixer 实验 (加分项) ---
declare -A -g variant_mixer_baseline=(
    [module]="src.mlp.train_mixer"
    [args]="--epochs 10 --batch_size 128 --lr 1e-3 --exp_name baseline"
)

# 定义执行单个实验变种的函数
run_variant() {
    local variant_name=$1
    shift
    local extra_args=$@

    # 使用间接引用获取变种的配置
    local module_ref="variant_$variant_name[module]"
    local args_ref="variant_$variant_name[args]"
    local module="${!module_ref}"
    local args="${!args_ref}"
    local log_file="logs/${variant_name}.log"

    if [ -z "$module" ]; then
        echo "错误: 未知的实验变种名称 '$variant_name'"
        return 1
    fi

    echo "=============================================================================="
    echo "  准备执行实验变种: $variant_name"
    echo "=============================================================================="
    echo "  - 训练模块: $module"
    echo "  - 默认参数: $args"
    echo "  - 日志文件: $log_file"
    if [ -n "$extra_args" ]; then
        echo "  - 覆盖参数: $extra_args"
    fi

    # --- 特殊情况: RNN姓氏分类实验不支持分布式训练 ---
    if [[ "$variant_name" == *"rnn_"* && "$variant_name" == *"_surname"* ]]; then
        echo "  - 检测到RNN姓氏实验，强制使用单进程模式。"
        {
            python -m "$module" $args $extra_args
        } 2>&1 | tee "$log_file"
    else
        # --- 标准分布式训练 ---
        if [ "$GPU_COUNT" -gt 0 ]; then
          echo "  - 使用GPU数量: $NPROC_PER_NODE"
        else
          echo "  - 使用CPU"
        fi
        echo "------------------------------------------------------------------------------"

        # 使用 torchrun 执行训练, -m 表示以模块方式运行
        # 将标准输出和标准错误都通过 tee 同时输出到控制台和日志文件
        # --master_port 是为了避免多个任务同时运行时端口冲突
        {
            torchrun --nproc_per_node=$NPROC_PER_NODE \
                --master_port=$((29500 + RANDOM % 100)) \
                -m "$module" \
                $args \
                $extra_args
        } 2>&1 | tee "$log_file"
    fi

    echo "------------------------------------------------------------------------------"
    echo "  实验变种 $variant_name 完成！"
    echo "=============================================================================="
    echo
}

# 主逻辑
if [ $# -gt 0 ]; then
    target=$1
    shift
    # 检查是运行整个实验类型还是单个变种
    if declare -p "variant_${target}" &> /dev/null; then
        # 是一个明确的变种名称
        run_variant "$target" "$@"
    else
        # 可能是实验类型 (mlp, cnn, etc.)
        echo "运行所有 '$target' 类型的实验变种..."
        for var_name in $(declare -p | grep "declare -A variant_${target}" | sed -e 's/declare -A variant_//' -e 's/=.*//'); do
            run_variant "$var_name" "$@"
        done
    fi
else
    # 如果没有提供参数，则按顺序运行所有实验的所有变种
    echo "未指定特定实验，将按默认配置运行所有实验变种..."
    all_variants=$(declare -p | grep "declare -A variant_" | sed -e 's/declare -A variant_//' -e 's/=.*//' | sort)
    for var_name in $all_variants; do
        run_variant "$var_name"
    done
fi

echo "=============================================================================="
echo "所有指定的实验均已执行完毕！"
echo "==============================================================================" 