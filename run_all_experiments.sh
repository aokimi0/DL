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

set -e
set -o pipefail

source /root/data-tmp/miniconda3/etc/profile.d/conda.sh
conda activate llm

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



if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 命令未找到。请确保 NVIDIA 驱动已正确安装。"
    exit 1
fi
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "警告: 未检测到GPU。将使用CPU进行训练。"
    NPROC_PER_NODE=1
else
    echo "检测到 $GPU_COUNT 个GPU。"
    NPROC_PER_NODE=$GPU_COUNT
fi
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))

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

# --- MLP 实验 (共5个变种) ---
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
declare -A -g variant_mixer_baseline=(
    [module]="src.mlp.train_mixer"
    [args]="--epochs 10 --batch_size 128 --lr 1e-3 --exp_name baseline"
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

# --- RNN 实验 (共3个变种) ---
declare -A -g variant_rnn_manual_surname=(
    [module]="src.rnn.train"
    [args]="--model manual_rnn --exp_name surname_manual_rnn --n_iters 100000 --print_every 4000 --plot_every 4000 --batch_size 256 --clip 1 --lr 0.0005"
    [run_style]="single_process"
)
declare -A -g variant_rnn_lstm_surname=(
    [module]="src.rnn.train"
    [args]="--model nn_lstm --exp_name surname_lstm --epochs 80 --batch_size 512 --clip 1 --lr 0.001"
    [run_style]="single_process"
)
declare -A -g variant_rnn_gru_surname=(
    [module]="src.rnn.train"
    [args]="--model nn_gru --exp_name surname_gru --epochs 80 --batch_size 512 --clip 1 --lr 0.001"
    [run_style]="single_process"
)

# --- GAN 实验 (共2个变种) ---
declare -A -g variant_gan_dcgan=(
    [module]="src.gan.train"
    [args]="--epochs 25 --batch_size 256 --exp_name dcgan"
)
declare -A -g variant_gan_dcgan_fashion_mnist=(
    [module]="src.gan.train"
    [args]="--exp_name gan_dcgan_fashion_mnist --dataset fashion_mnist --epochs 15 --batch_size 64"
)



run_variant() {
    local variant_name=$1
    shift
    local extra_args=$@

    local module_ref="variant_$variant_name[module]"
    local args_ref="variant_$variant_name[args]"
    local run_style_ref="variant_$variant_name[run_style]"
    local module="${!module_ref}"
    local args="${!args_ref}"
    local run_style="${!run_style_ref}"
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

    # 根据 run_style 智能选择执行方式
    local cmd_to_run
    if [[ "$run_style" == "single_process" ]]; then
        echo "  - 运行方式: 单进程 (python -m)"
        # 对于单进程模式，我们直接使用 python 命令。
        # 脚本内部的 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 会自动选择设备。
        cmd_to_run="python -m $module $args $extra_args"
    else
        if [ "$GPU_COUNT" -gt 0 ]; then
            echo "  - 运行方式: 多进程 (torchrun)"
            echo "  - 使用GPU数量: $NPROC_PER_NODE"
            # --master_port 是为了避免多个任务同时运行时端口冲突
            cmd_to_run="torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$((29500 + RANDOM % 100)) -m $module $args $extra_args"
        else
            echo "  - 运行方式: 单进程 (python -m, 无GPU)"
            cmd_to_run="python -m $module $args $extra_args"
        fi
    fi

    echo "------------------------------------------------------------------------------"

    # 执行最终构建的命令
    # 将标准输出和标准错误都通过 tee 同时输出到控制台和日志文件
    {
        eval $cmd_to_run
    } 2>&1 | tee "$log_file"


    echo "------------------------------------------------------------------------------"
    echo "  实验变种 $variant_name 完成！"
    echo "=============================================================================="
    echo
}

if [ $# -gt 0 ]; then
    target=$1
    shift
    if declare -p "variant_${target}" &> /dev/null; then
        run_variant "$target" "$@"
    else
        echo "运行所有 '$target' 类型的实验变种..."
        for var_name in $(declare -p | grep "declare -A variant_${target}" | sed -e 's/declare -A variant_//' -e 's/=.*//'); do
            run_variant "$var_name" "$@"
        done
    fi
else
    echo "未指定特定实验，将按默认配置运行所有实验变种..."
    all_variants=$(declare -p | grep "declare -A variant_" | sed -e 's/declare -A variant_//' -e 's/=.*//' | sort)
    for var_name in $all_variants; do
        run_variant "$var_name"
    done
fi

echo "=============================================================================="
echo "所有指定的实验均已执行完毕！"
echo "==============================================================================" 