import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# PRUNABLE LINEAR LAYER
class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, temperature: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        pruned_weights = self.weight * gates
        return torch.matmul(x, pruned_weights.t()) + self.bias

    def get_gates(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores / self.temperature)

# NEURAL NETWORK
class PrunableNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(32 * 32 * 3, 1024),
            nn.ReLU(),
            PrunableLinear(1024, 512),
            nn.ReLU(),
            PrunableLinear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   
        return self.net(x)

# DATASET 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
)

# SPARSITY REGULARISATION LOSS 
def compute_sparsity_loss(model: nn.Module) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    total_gates = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores / module.temperature)
            total_loss = total_loss + gates.sum()       
            total_gates += gates.numel()
    return total_loss / total_gates

# TRAINING
def train_model(lambda_val: float, epochs: int = 20) -> nn.Module:
    model = PrunableNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            ce_loss = criterion(logits, labels)
            sparsity_loss = compute_sparsity_loss(model)
            loss = ce_loss + lambda_val * (sparsity_loss * 1000)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        print(f"  λ={lambda_val} | Epoch {epoch + 1:>2}/{epochs} | Loss: {avg_loss:.4f}")

    return model

# EVALUATION 
def evaluate(model: nn.Module) -> float:
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total

# SPARSITY MEASUREMENT
def calculate_sparsity(model: nn.Module, threshold: float = 1e-2) -> float:
    total = zero = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gates()
            total += gates.numel()
            zero += (gates < threshold).sum().item()

    return 100.0 * zero / total

lambda_values = [0.01, 0.1, 1.0]
results = []         

best_model = None
best_score = -float("inf")

for lam in lambda_values:
    print(f"\n{'='*60}")
    print(f" Training with λ = {lam}")
    print(f"{'='*60}")

    model = train_model(lam, epochs=20)
    acc = evaluate(model)
    sparsity = calculate_sparsity(model)

    # Persist the checkpoint
    ckpt_path = f"model_lambda_{lam}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  → Saved checkpoint: {ckpt_path}")

    score = acc - 0.1 * (100.0 - sparsity)   

    if score > best_score:
        best_score = score
        best_model = model
        best_lambda = lam

    results.append((lam, acc, sparsity))
    print(f"  → Test Accuracy: {acc:.2f}%  |  Sparsity: {sparsity:.2f}%")

# RESULTS
print("\n" + "="*55)
print(f"{'Lambda':>10} | {'Test Accuracy (%)':>18} | {'Sparsity (%)':>13}")
print("-"*55)
for lam, acc, spar in results:
    print(f"{lam:>10} | {acc:>18.2f} | {spar:>13.2f}")
print("="*55)
print(f"\nBest model selected at λ = {best_lambda}")
# GATE DISTRIBUTION PLOT 
all_gates = []
for module in best_model.modules():
    if isinstance(module, PrunableLinear):
        gates = module.get_gates().cpu().numpy().flatten()
        all_gates.extend(gates)

all_gates = np.array(all_gates)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(all_gates, bins=100, color="steelblue", edgecolor="none")
ax.set_yscale("log")
ax.set_title(f"Gate Value Distribution – Best Model (λ = {best_lambda})", fontsize=13)
ax.set_xlabel("Gate Value", fontsize=11)
ax.set_ylabel("Frequency (log scale)", fontsize=11)
ax.axvline(x=1e-2, color="red", linestyle="--", linewidth=1.2,
           label="Pruning threshold (1e-2)")
ax.legend()
plt.tight_layout()
plt.savefig("gate_distribution.png", dpi=150)
print("\nGate distribution plot saved → gate_distribution.png")
try:
    import google.colab  
    from google.colab import files
    files.download(f"model_lambda_{best_lambda}.pth")
except ImportError:
    pass
