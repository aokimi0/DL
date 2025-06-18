# 深度学习实验

本项目是一个结构化的深度学习实验仓库，旨在通过标准化的方式实现、训练和评估四个经典的深度学习模型：多层感知机 (MLP)、卷积神经网络 (CNN)、循环神经网络 (RNN) 和生成对抗网络 (GAN)。

## ✨ 主要特性

- **模块化设计**: 每个实验（MLP, CNN, RNN, GAN）都有独立的模块，代码结构清晰。
- **配置驱动**: 所有实验超参数均在 `run_all_experiments.sh` 中集中定义，易于管理和复现。
- **自动化执行**: 通过一个脚本即可运行所有实验，或指定单个实验类型/变种。
- **分布式训练**: 默认使用 `torchrun` 和所有可用 GPU 进行分布式训练，以加速实验进程。
- **标准化输出**: 自动将训练日志保存到 `logs/` 目录，将生成的可视化图表保存到 `fig/` 目录。

## 🛠️ 技术栈

- **Python**: 3.12+
- **核心框架**: PyTorch, Torchvision
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib

## 📂 目录结构

```
.
├── data/              # 存放数据集 (自动下载)
├── docs/              # 存放项目文档和原始需求
├── fig/               # 存放所有实验生成的可视化图表
├── logs/              # 存放每个实验变种的运行日志
├── report/            # 存放 Markdown/Pdf 格式的实验报告
├── src/               # 存放所有 Python 源代码
├── .gitignore         # Git 忽略文件配置
├── README.md          # 本项目说明文件
├── requirements.txt   # Python 依赖包列表
└── run_all_experiments.sh # 核心实验执行脚本
```

## 🚀 快速开始

### 1. 环境配置

首先，克隆本仓库到您的本地机器：
```bash
git clone <your-repository-url>
cd <repository-name>
```

推荐使用 Conda 创建并激活一个独立的 Python 环境：
```bash
conda create --name dl-exp python=3.12 -y
conda activate dl-exp
```

然后，安装所有必需的依赖项：
```bash
pip install -r requirements.txt
```

脚本会自动检查并尝试安装绘图所需的中文字体。如果失败，您可能需要根据您的操作系统手动安装。

### 2. 执行实验

本项目使用 `run_all_experiments.sh` 脚本来管理和执行所有实验。该脚本提供了灵活的执行方式。

**请确保在项目根目录下执行以下所有命令。**

#### a) 运行所有实验

执行此命令将按顺序运行 `run_all_experiments.sh` 中定义的所有实验变种。
```bash
bash run_all_experiments.sh
```

#### b) 运行特定类型的所有实验

您可以指定一个实验类型（如 `cnn`, `mlp`, `rnn`, `gan`）来运行该类型下的所有变种。
```bash
# 仅运行所有 CNN 相关的实验
bash run_all_experiments.sh cnn
```

#### c) 运行单个实验变种

您可以直接运行一个具体的实验变种。变种名称在 `run_all_experiments.sh` 中定义（例如 `cnn_resnet18_adam`）。
```bash
# 仅运行 ResNet18 + Adam 的 CNN 实验
bash run_all_experiments.sh cnn_resnet18_adam
```

#### d) 覆盖默认参数

在运行单个实验变种时，您可以从命令行覆盖其默认参数。
```bash
# 运行 cnn_resnet18_adam 变种，但将训练轮数修改为 20
bash run_all_experiments.sh cnn_resnet18_adam --epochs 20
```

## 📊 查看结果

- **日志**: 每个实验变种的完整控制台输出都会被重定向到 `logs/<variant_name>.log` 文件中。
- **图表**: 实验过程中生成的图表（如损失曲线、模型性能图等）会根据实验名称保存在 `fig/<exp_type>/<exp_name>/` 目录下。
- **报告**: 对每个实验的详细分析、结果和结论都记录在 `report/` 目录下的相应 Markdown 文件中。

## 🤝 贡献

欢迎通过 Pull Request 或 Issue 对本项目进行改进。 