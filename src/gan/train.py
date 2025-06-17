import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from src.gan.models.dcgan import Generator, Discriminator
from src.gan.data_loader import get_mnist_loader, get_fashion_mnist_loader
from src.utils import setup_plotting, save_model, create_exp_dir


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_gan(
    rank: int,
    epochs: int,
    latent_dim: int,
    batch_size: int,
    lr: float,
    b1: float,
    b2: float,
    exp_name: str,
    dataset: str = "mnist",
):
    if rank == 0:
        print(f"实验 '{exp_name}' 的目录已创建。")
        _, fig_dir, out_dir = create_exp_dir(exp_name, "gan")

    if dataset == "mnist":
        train_dataset = get_mnist_loader(batch_size, return_dataset=True)
    elif dataset == "fashion_mnist":
        train_dataset = get_fashion_mnist_loader(batch_size, return_dataset=True)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

    sampler = DistributedSampler(train_dataset)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    if rank == 0:
        print(f"已加载 {dataset.upper()} 数据集。")

    generator = Generator(latent_dim).to(rank)
    discriminator = Discriminator().to(rank)
    
    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    adversarial_loss = nn.BCELoss().to(rank)

    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(b1, b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(b1, b2)
    )

    if rank == 0:
        print("开始分布式训练循环...")
    
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=rank)

    for epoch in range(epochs):
        dataloader.sampler.set_epoch(epoch)
        for i, (imgs, _) in enumerate(dataloader):
            valid = torch.full((imgs.size(0), 1), 1.0, device=rank, requires_grad=False)
            fake = torch.full((imgs.size(0), 1), 0.0, device=rank, requires_grad=False)
            
            real_imgs = imgs.to(rank)

            optimizer_D.zero_grad()

            with discriminator.no_sync():
                real_loss = adversarial_loss(discriminator(real_imgs).view(-1, 1), valid)
                real_loss.backward()

            z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=rank)
            gen_imgs = generator(z).detach()
            fake_loss = adversarial_loss(discriminator(gen_imgs).view(-1, 1), fake)
            fake_loss.backward()

            d_loss = (real_loss + fake_loss) / 2
            
            optimizer_D.step()

            optimizer_G.zero_grad()

            z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=rank)
            gen_imgs = generator(z)

            g_loss = adversarial_loss(discriminator(gen_imgs).view(-1, 1), valid)

            g_loss.backward()
            optimizer_G.step()

            if rank == 0 and i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
        
        if rank == 0:
            with torch.no_grad():
                gen_imgs = generator.module(fixed_noise).detach().cpu()
            save_image(gen_imgs, f"{fig_dir}/{epoch}.png", nrow=8, normalize=True)
            save_model(generator.module, f"{out_dir}/generator.pth")
            save_model(discriminator.module, f"{out_dir}/discriminator.pth")

    if rank == 0:
        print(f"训练完成。模型已保存至 {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="使用DDP训练一个DCGAN模型")
    parser.add_argument("--exp_name", type=str, required=True, help="实验名称")
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="每个GPU上的批处理大小")
    parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
    parser.add_argument("--latent_dim", type=int, default=100, help="潜在空间维度")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam优化器的beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam优化器的beta2")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="mnist", 
        choices=["mnist", "fashion_mnist"],
        help="使用的数据集 (mnist 或 fashion_mnist)"
    )
    args = parser.parse_args()

    local_rank = setup_distributed()

    train_gan(
        rank=local_rank,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        b1=args.b1,
        b2=args.b2,
        exp_name=args.exp_name,
        dataset=args.dataset,
    )

    cleanup_distributed()


if __name__ == "__main__":
    main()
