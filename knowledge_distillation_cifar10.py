
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
    batch_size  = 256,          # change accordingly with gpu memory
    num_workers = 4,
    epochs      = 20,           # 20 for fast prototyping
    lr          = 0.1,
    momentum    = 0.9,
    weight_decay= 5e-4,
    # 蒸馏超参
    T           = 4.0,          # Temperature
    alpha       = 0.7,          # Loss
    seed        = 42,
)

torch.manual_seed(CFG["seed"])
torch.backends.cudnn.benchmark = True  
DEVICE = CFG["device"]
print(f"使用设备: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


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
#  Model definition
#  
# ─────────────────────────────────────────────
def make_resnet50_teacher(num_classes=10):
    """Teacher: ResNet-50，~ 150M parameters"""
    model = models.resnet50(weights=None)
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_resnet18_student(num_classes=10):
    """Student: ResNet-18，~ 11M parameters"""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ─────────────────────────────────────────────
#  Loss function
#  L_total = alpha * L_KL(soft) + (1-alpha) * L_CE(hard) 
# ─────────────────────────────────────────────
def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    loss_kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    
    loss_ce = F.cross_entropy(student_logits, true_labels)

    return alpha * loss_kl + (1.0 - alpha) * loss_ce

def train_standard(model, loader, optimizer, epoch):
    """Standard train（For Teacher and Student）"""
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
    """Distillation train"""
    student.train()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"  Distill Epoch {epoch}", leave=False)
    with torch.no_grad():
        pass  
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
#  Main experiment
# ─────────────────────────────────────────────
def run_experiment():
    epochs = CFG["epochs"]
    results = {}

    # ── 阶段 1: Pre-train ────────────────────
    print("\n" + "="*55)
    print("Stage 1 / 3  Teacher (ResNet-50) Pre-train")
    print("="*55)
    teacher = make_resnet50_teacher().to(DEVICE)
    print(f"Teacher parameters: {count_params(teacher)/1e6:.1f}M")

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
    print(f"Teacher final test acc: {100*teacher_acc:.2f}%  ({time.time()-t0:.0f}s)")
    results["teacher"] = teacher_acc
    torch.save(teacher.state_dict(), "teacher_resnet50.pth")

    # ── Stage 2: Distillation Student ───────────────
    print("\n" + "="*55)
    print("Stage 2 / 3  Student (ResNet-18) knowledge distillation")
    print(f"  T={CFG['T']}, alpha={CFG['alpha']}")
    print("="*55)
    student_kd = make_resnet18_student().to(DEVICE)
    print(f"Student parameters: {count_params(student_kd)/1e6:.1f}M")

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
    print(f"Student(KD) final test acc: {100*kd_acc:.2f}%  ({time.time()-t0:.0f}s)")
    results["student_kd"] = kd_acc

    # ── Stage 3:  Student (no distillation，for comparison) ────
    print("\n" + "="*55)
    print("阶段 3 / 3  Student (ResNet-18) ")
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
    print(f"Student(base) final test acc: {100*base_acc:.2f}%  ({time.time()-t0:.0f}s)")
    results["student_base"] = base_acc

    # ── Conclusion ──────────────────────────────────
    print("\n" + "="*55)
    print("Conclusion")
    print("="*55)
    print(f"  Teacher  (ResNet-50, 150M):          {100*results['teacher']:.2f}%")
    print(f"  Student  (ResNet-18,  11M, no distillation): {100*results['student_base']:.2f}%")
    print(f"  Student  (ResNet-18,  11M, distilled):   {100*results['student_kd']:.2f}%")
    print(f"\n  Distilled vs non-distilled: +{100*(results['student_kd']-results['student_base']):.2f}%")
    print(f"  Teacher Acc loss: -{100*(results['teacher']-results['student_kd']):.2f}%")
    print(f"  Parameter compression ratio: {count_params(teacher)/count_params(student_kd):.1f}×")
    print("="*55)

    return results

if __name__ == "__main__":
    run_experiment()
