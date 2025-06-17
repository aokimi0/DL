import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from src.gan.models.dcgan import Generator
from src.utils import setup_plotting

def generate_random_images(generator, latent_dim, num_images, output_path, device):
    print(f"正在生成 {num_images} 张随机图像...")
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        fake_images = generator(noise).cpu()
        save_image(fake_images, output_path, nrow=num_images, normalize=True)
    print(f"随机图像已保存至: {output_path}")


def generate_and_save_manipulation_images(
    generator,
    latent_dim,
    base_vector_indices,
    dims_to_tweak,
    tweak_strengths,
    output_dir,
    device,
):
    print("开始执行潜在空间维度操控分析...")
    torch.manual_seed(42)
    base_latent_vectors = torch.randn(100, latent_dim, 1, 1, device=device)
    print(f"已生成100个基准潜在向量。将分析索引为 {base_vector_indices} 的向量。")

    os.makedirs(output_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        for vec_idx in base_vector_indices:
            base_vector = base_latent_vectors[vec_idx:vec_idx+1]
            print(f"\n正在操控基准向量: {vec_idx}...")

            for dim_to_tweak in dims_to_tweak:
                manipulated_vectors = []
                for strength in tweak_strengths:
                    tweaked_vector = base_vector.clone()
                    tweaked_vector[0, dim_to_tweak, 0, 0] += strength
                    manipulated_vectors.append(tweaked_vector)
                
                all_tweaked_vectors = torch.cat(manipulated_vectors, dim=0)

                fake_images = generator(all_tweaked_vectors).cpu()

                grid = make_grid(fake_images, nrow=8, normalize=True)
                
                plt.figure(figsize=(12, 2), dpi=150)
                plt.imshow(np.transpose(grid, (1, 2, 0)))
                plt.axis("off")
                title = (f"基准向量 {vec_idx}, 调整维度 {dim_to_tweak}")
                plt.title(title, fontsize=10)
                
                output_path = os.path.join(output_dir, f"manipulation_vec{vec_idx}_dim{dim_to_tweak}.png")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()

    print(f"\n所有维度操控分析图像已保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="从预训练的DCGAN模型生成图像并进行维度操控分析")
    parser.add_argument(
        "--model_path",
        type=str,
        default="out/gan/gan_dcgan_fashion_mnist/generator.pth",
        help="预训练生成器模型的路径",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="gan_dcgan_fashion_mnist",
        help="实验名称，用于确定输出目录"
    )
    parser.add_argument("--latent_dim", type=int, default=100, help="潜在空间维度")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="禁用CUDA")
    args = parser.parse_args()

    output_dir = os.path.join("fig/gan", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")

    setup_plotting()

    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件未找到于 '{args.model_path}'")
        print("请先运行GAN训练脚本 `run_all_experiments.sh gan_dcgan_fashion_mnist` 来生成模型文件。")
        return

    generator = Generator(nz=args.latent_dim).to(device)
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        generator.load_state_dict(state_dict)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
        
    print(f"成功从 '{args.model_path}' 加载模型。")

    generate_random_images(
        generator=generator,
        latent_dim=args.latent_dim,
        num_images=8,
        output_path=os.path.join(output_dir, "random_generation.png"),
        device=device,
    )

    base_vector_indices = [5, 18, 33, 42, 99] 
    dims_to_tweak = [5, 15, 25]
    tweak_strengths = list(np.linspace(-4.0, 4.0, 8))

    generate_and_save_manipulation_images(
        generator=generator,
        latent_dim=args.latent_dim,
        base_vector_indices=base_vector_indices,
        dims_to_tweak=dims_to_tweak,
        tweak_strengths=tweak_strengths,
        output_dir=output_dir,
        device=device,
    )

if __name__ == "__main__":
    main() 