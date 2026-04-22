# Self-Pruning Neural Network

This project implements a neural network that learns to prune its own weights during training using learnable gates. The goal is to reduce unnecessary connections while maintaining good performance.

---

## Approach

Each weight in the network is multiplied by a learnable gate value between 0 and 1. These gates are obtained using a sigmoid function applied to trainable parameters.

During training, an additional sparsity loss is added to the standard classification loss. This encourages many gate values to move toward zero, effectively removing less important connections.

---

## Why L1 Regularization Encourages Sparsity

L1 regularization penalizes the absolute value of the gate parameters. Unlike L2, its gradient does not vanish as values approach zero. This creates a constant push toward zero, making it effective at driving many gates to exactly zero and producing a sparse network.

---

## Results

I trained the model with three different values of λ to observe how sparsity affects performance.

For λ = 0.01, the model achieved around 56.9% accuracy with about 10% sparsity, meaning only a small portion of weights were pruned.

When λ was increased to 0.1, sparsity increased significantly to around 70%, and accuracy slightly improved to about 57.2%. This suggests that moderate pruning helped remove unnecessary connections and improved generalization.

At λ = 1.0, the network became extremely sparse (around 99%), but accuracy dropped to about 54.6%, indicating that excessive pruning starts to harm performance.

Overall, the results show a clear trade-off between sparsity and accuracy.

---

## Observations

Initially, sparsity loss had little effect because it was too small compared to the classification loss. After adjusting its scale, the model began to effectively prune connections.

Interestingly, moderate sparsity improved performance, likely because it acted as a form of regularization by removing noisy or less useful weights.

However, too much sparsity led to over-pruning, where important connections were also removed, causing performance to drop.

---

## Gate Distribution

The distribution of gate values shows that most values are pushed very close to zero, while a small number remain near one. This indicates that the model successfully learned to make near-binary decisions about which connections to keep or remove.

![Gate Distribution](results.png)

---

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the script:

python main.py

---

## Key Insight

Moderate sparsity improves generalization, while excessive pruning degrades performance. This demonstrates the importance of balancing model efficiency and accuracy.
