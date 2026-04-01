"""
知识蒸馏实验 — CIFAR-10
Teacher: ResNet-50 (~150M params)  →  Student: ResNet-18 (~11M params)
硬件目标: RTX 5070 Ti 16GB

运行前安装依赖:
    pip install torch torchvision tqdm

运行方式:
    python knowledge_distillation_cifar10.py

大约时间 (5070 Ti):
    Teacher 预训练  ~15 min (20 epochs)
    Student 蒸馏    ~8  min (20 epochs)
    Student 基线    ~8  min (对照组，不使用蒸馏)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# ─────────────────────────────────────────────
#  配置
# ─────────────────────────────────────────────
CFG = dict(
    device      = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size  = 256,          # 5070 Ti 16GB 完全够用，可调到 512
    num_workers = 4,
    epochs      = 20,           # 快速验证；正式实验可改 100
    lr          = 0.1,
    momentum    = 0.9,
    weight_decay= 5e-4,
    # 蒸馏超参
    T           = 4.0,          # 温度：控制软标签软化程度，推荐 3-6
    alpha       = 0.7,          # 蒸馏损失权重；(1-alpha) 分给真实标签损失
    seed        = 42,
)

torch.manual_seed(CFG["seed"])
torch.backends.cudnn.benchmark = True  # 对固定输入尺寸有加速
DEVICE = CFG["device"]
print(f"使用设备: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
#  数据集  (CIFAR-10, 自动下载到 ./data)
# ─────────────────────────────────────────────
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=train_transform)
test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

# ─────────────────────────────────────────────
#  模型定义
#  CIFAR-10 图像是 32×32，需要把 ResNet 的第一层卷积改小
# ─────────────────────────────────────────────
def make_resnet50_teacher(num_classes=10):
    """Teacher: ResNet-50，约 150M 参数"""
    model = models.resnet50(weights=None)
    # 适配 32×32 输入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_resnet18_student(num_classes=10):
    """Student: ResNet-18，约 11M 参数"""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ─────────────────────────────────────────────
#  蒸馏损失函数
#  L_total = alpha * L_KL(soft) + (1-alpha) * L_CE(hard)
#  注意：KL散度要乘以 T^2 以补偿梯度幅度
# ─────────────────────────────────────────────
def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    # 软标签损失（KL散度）
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    loss_kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

    # 真实标签损失（交叉熵）
    loss_ce = F.cross_entropy(student_logits, true_labels)

    return alpha * loss_kl + (1.0 - alpha) * loss_ce

# ─────────────────────────────────────────────
#  通用训练 / 评估函数
# ─────────────────────────────────────────────
def train_standard(model, loader, optimizer, epoch):
    """标准训练（用于 Teacher 和基线 Student）"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"  Train Epoch {epoch}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.3f}", acc=f"{100*correct/total:.1f}%")
    return total_loss / total, correct / total

def train_distill(student, teacher, loader, optimizer, epoch, T, alpha):
    """蒸馏训练"""
    student.train()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"  Distill Epoch {epoch}", leave=False)
    with torch.no_grad():
        pass  # teacher 不需要梯度，但要在 forward 时关闭
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        s_logits = student(imgs)
        with torch.no_grad():
            t_logits = teacher(imgs)
        loss = distillation_loss(s_logits, t_logits, labels, T, alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (s_logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.3f}", acc=f"{100*correct/total:.1f}%")
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─────────────────────────────────────────────
#  实验主流程
# ─────────────────────────────────────────────
def run_experiment():
    epochs = CFG["epochs"]
    results = {}

    # ── 阶段 1: 训练 Teacher ────────────────────
    print("\n" + "="*55)
    print("阶段 1 / 3  Teacher (ResNet-50) 预训练")
    print("="*55)
    teacher = make_resnet50_teacher().to(DEVICE)
    print(f"Teacher 参数量: {count_params(teacher)/1e6:.1f}M")

    opt_t = optim.SGD(teacher.parameters(), lr=CFG["lr"],
                      momentum=CFG["momentum"], weight_decay=CFG["weight_decay"])
    sched_t = CosineAnnealingLR(opt_t, T_max=epochs)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_standard(teacher, train_loader, opt_t, ep)
        sched_t.step()
        if ep % 5 == 0 or ep == epochs:
            acc = evaluate(teacher, test_loader)
            print(f"  Epoch {ep:3d}/{epochs}  Teacher test acc: {100*acc:.2f}%")
    teacher_acc = evaluate(teacher, test_loader)
    print(f"Teacher 最终 test acc: {100*teacher_acc:.2f}%  ({time.time()-t0:.0f}s)")
    results["teacher"] = teacher_acc
    torch.save(teacher.state_dict(), "teacher_resnet50.pth")

    # ── 阶段 2: 蒸馏训练 Student ───────────────
    print("\n" + "="*55)
    print("阶段 2 / 3  Student (ResNet-18) 知识蒸馏")
    print(f"  T={CFG['T']}, alpha={CFG['alpha']}")
    print("="*55)
    student_kd = make_resnet18_student().to(DEVICE)
    print(f"Student 参数量: {count_params(student_kd)/1e6:.1f}M")

    opt_s = optim.SGD(student_kd.parameters(), lr=CFG["lr"],
                      momentum=CFG["momentum"], weight_decay=CFG["weight_decay"])
    sched_s = CosineAnnealingLR(opt_s, T_max=epochs)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_distill(student_kd, teacher, train_loader, opt_s, ep, CFG["T"], CFG["alpha"])
        sched_s.step()
        if ep % 5 == 0 or ep == epochs:
            acc = evaluate(student_kd, test_loader)
            print(f"  Epoch {ep:3d}/{epochs}  Student(KD) test acc: {100*acc:.2f}%")
    kd_acc = evaluate(student_kd, test_loader)
    print(f"Student(KD) 最终 test acc: {100*kd_acc:.2f}%  ({time.time()-t0:.0f}s)")
    results["student_kd"] = kd_acc

    # ── 阶段 3: 基线 Student (无蒸馏，对照组) ────
    print("\n" + "="*55)
    print("阶段 3 / 3  Student (ResNet-18) 基线 (无蒸馏)")
    print("="*55)
    student_base = make_resnet18_student().to(DEVICE)

    opt_b = optim.SGD(student_base.parameters(), lr=CFG["lr"],
                      momentum=CFG["momentum"], weight_decay=CFG["weight_decay"])
    sched_b = CosineAnnealingLR(opt_b, T_max=epochs)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_standard(student_base, train_loader, opt_b, ep)
        sched_b.step()
        if ep % 5 == 0 or ep == epochs:
            acc = evaluate(student_base, test_loader)
            print(f"  Epoch {ep:3d}/{epochs}  Student(base) test acc: {100*acc:.2f}%")
    base_acc = evaluate(student_base, test_loader)
    print(f"Student(base) 最终 test acc: {100*base_acc:.2f}%  ({time.time()-t0:.0f}s)")
    results["student_base"] = base_acc

    # ── 汇总 ──────────────────────────────────
    print("\n" + "="*55)
    print("实验结果汇总")
    print("="*55)
    print(f"  Teacher  (ResNet-50, 150M):          {100*results['teacher']:.2f}%")
    print(f"  Student  (ResNet-18,  11M, 无蒸馏): {100*results['student_base']:.2f}%")
    print(f"  Student  (ResNet-18,  11M, 蒸馏):   {100*results['student_kd']:.2f}%")
    print(f"\n  蒸馏增益 vs 基线: +{100*(results['student_kd']-results['student_base']):.2f}%")
    print(f"  Teacher 精度损失: -{100*(results['teacher']-results['student_kd']):.2f}%")
    print(f"  参数压缩比: {count_params(teacher)/count_params(student_kd):.1f}×")
    print("="*55)

    return results

if __name__ == "__main__":
    run_experiment()
