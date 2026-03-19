# fed-vit-autorl

Federated learning of Vision Transformers with automated RL-based hyperparameter optimization.

## What it does

Train a Vision Transformer across multiple clients without sharing raw data, while a UCB1 multi-armed bandit automatically tunes learning rate and batch size each federation round.

```
Round 1: UCB1 selects lr=1e-04, bs=16  → federated acc: 12.7%
Round 2: UCB1 selects lr=1e-04, bs=32  → federated acc: 20.3%
Round 3: UCB1 selects lr=5e-04, bs=16  → federated acc: 34.3%
Round 4: UCB1 selects lr=5e-04, bs=32  → federated acc: 44.0%
Round 5: UCB1 selects lr=1e-03, bs=16  → federated acc: 59.7%  ← best
```

## Architecture

### `TinyViT`
Small Vision Transformer (~115K parameters):
- Patch embedding: Conv2D projecting 8×8 patches to 64-dim tokens
- 3 transformer blocks: LayerNorm → Multi-Head Self-Attention → MLP (GELU)
- CLS token classification head

### `FederatedViTClient`
- Receives global model from server (deep copy)
- Trains locally for N epochs with Adam optimizer
- Computes weight delta = local − global
- Applies **top-k gradient sparsification** (keeps top 10% by magnitude) before upload
- Compression ratio: ~10× reduction in communication cost

### `FederatedViTServer`
- Distributes global model to all clients each round
- Aggregates sparse updates via **FedAvg** (weighted by dataset size)
- Applies **server-side momentum** to the aggregated update (reduces oscillation)

### `RLHyperparamTuner`
**UCB1 multi-armed bandit** over a grid of (lr × batch_size) arms:
- Explores arms with high uncertainty: score = mean_reward + c·√(ln(t)/n)
- Forces exploration of untried arms first (UCB score = ∞)
- Updates reward with federated accuracy after each round
- Reports best configuration after all rounds

### `SyntheticImageDataset`
Generates 32×32 RGB images of colored geometric shapes:
- **5 shapes**: circle, square, triangle, diamond, cross
- **5 colors**: red, green, blue, yellow, purple
- **25 classes** total (shape × color)
- Deterministic generation (seeded per sample)
- Supports client sharding for federated experiments

## Quick start

```bash
# Run the demo (3 clients, 5 rounds, UCB1 tuning)
python demo.py
```

Requires Python 3.9+ with PyTorch.

## Install

```bash
pip install -e .
```

## Usage

```python
from fed_vit_autorl import (
    TinyViT, SyntheticImageDataset,
    FederatedViTClient, FederatedViTServer,
    RLHyperparamTuner
)

# Model
model = TinyViT(img_size=32, patch_size=8, num_classes=25, embed_dim=64, depth=3)
print(f"Parameters: {model.num_parameters():,}")  # ~115K

# Data (sharded across 3 clients)
train_data = [SyntheticImageDataset(1500, client_id=i, num_clients=3) for i in range(3)]
test_data = SyntheticImageDataset(300, seed=999)

# Server + clients
server = FederatedViTServer(model, momentum=0.9)
clients = [FederatedViTClient(i, train_data[i], sparsity_k=0.1) for i in range(3)]
tuner = RLHyperparamTuner(lr_options=[1e-4, 1e-3, 1e-2], batch_size_options=[16, 32])

# Federation loop
for round_num in range(10):
    lr, batch_size, arm_idx = tuner.select()  # UCB1 picks hyperparams
    global_model = server.get_global_model()

    all_updates = []
    for client in clients:
        client.set_model(global_model)
        client.train(lr=lr, batch_size=batch_size, epochs=3)
        all_updates.append(client.get_sparse_updates(global_model))

    server.aggregate(all_updates)
    acc = server.evaluate(test_data)
    tuner.update(arm_idx, reward=acc)
    print(f"Round {round_num+1}: acc={acc:.4f} (lr={lr:.0e}, bs={batch_size})")

print(tuner.summary())
```

## Tests

```bash
python -m pytest tests/ -v
```

50 tests covering model correctness, dataset generation, federated aggregation, gradient sparsification, and UCB1 bandit behavior.

## Project structure

```
fed_vit_autorl/
  __init__.py      — package exports
  models.py        — TinyViT (PatchEmbedding, TransformerBlock)
  data.py          — SyntheticImageDataset
  federated.py     — FederatedViTClient, FederatedViTServer, sparse gradients
  rl_tuner.py      — RLHyperparamTuner (UCB1)
tests/
  test_models.py
  test_data.py
  test_federated.py
  test_rl_tuner.py
demo.py            — end-to-end demo
```

## License

MIT
