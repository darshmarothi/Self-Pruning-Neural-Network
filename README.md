# Self-Pruning Neural Network

## Overview
This project implements a neural network that learns to prune itself during training using learnable gating parameters.

Each weight is associated with a sigmoid gate that determines its importance. A sparsity regularization term encourages unnecessary connections to be removed.

---

## Methodology

### Prunable Layer
- Each weight has a learnable gate
- Gate = sigmoid(gate_score)
- Effective weight = weight * gate

### Loss Function
Total Loss = CrossEntropy + λ * Sparsity Loss

Sparsity loss:
- L1 penalty on sigmoid gate values
- Encourages gates to move toward 0

---

## Results

| Lambda | Accuracy | Sparsity (%) |
|--------|---------|--------------|
| 0.01   | 55.55%  | 66.45%       |
| 0.05   | 55.30%  | 68.73%       |
| 0.1    | 55.23%  | 70.65%       |

---

##  Gate Distribution
<img width="1564" height="1232" alt="image" src="https://github.com/user-attachments/assets/85adfc59-9c4a-437b-ad53-d818263dbc59" />


---

## Key Insight

Due to the continuous nature of sigmoid gating, weights are attenuated rather than forced to exact zero. Therefore, a threshold of 0.5 is used to determine effective pruning.

---

## Dataset

CIFAR-10 dataset from torchvision.

---

## How to Run

```bash
pip install torch torchvision matplotlib
python self_pruning_nn.py
