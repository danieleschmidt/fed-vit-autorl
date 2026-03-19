"""
Federated ViT + AutoRL Demo
===========================
3 clients, 5 federation rounds, UCB1 hyperparameter tuning.
Compares federated accuracy vs local-only baseline.
"""

import copy
import sys

import torch

from fed_vit_autorl import TinyViT, SyntheticImageDataset, FederatedViTClient, FederatedViTServer, RLHyperparamTuner


def print_banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def run_demo(
    num_clients: int = 3,
    num_rounds: int = 5,
    local_epochs: int = 3,
    device: str = "cpu",
    num_train: int = 900,
    num_test: int = 300,
):
    print_banner("fed-vit-autorl: Federated ViT + UCB1 AutoRL Demo")
    print(f"  Clients:       {num_clients}")
    print(f"  Rounds:        {num_rounds}")
    print(f"  Local epochs:  {local_epochs}")
    print(f"  Device:        {device}")
    print(f"  Train samples: {num_train}  (sharded across clients)")
    print(f"  Test samples:  {num_test}")

    # --- Data ---
    print_banner("Generating synthetic datasets")
    client_datasets = [
        SyntheticImageDataset(num_samples=num_train, seed=42 + i, client_id=i, num_clients=num_clients)
        for i in range(num_clients)
    ]
    test_dataset = SyntheticImageDataset(num_samples=num_test, seed=999)
    for i, ds in enumerate(client_datasets):
        print(f"  Client {i}: {len(ds)} samples")
    print(f"  Test set: {len(test_dataset)} samples, {test_dataset.num_classes()} classes")

    # --- Model ---
    print_banner("Initializing TinyViT")
    model = TinyViT(
        img_size=32,
        patch_size=8,
        num_classes=SyntheticImageDataset.num_classes(),
        embed_dim=64,
        depth=3,
        num_heads=4,
    )
    print(f"  Parameters: {model.num_parameters():,}")

    # --- Federated setup ---
    server = FederatedViTServer(model, device=device, momentum=0.9)
    clients = [
        FederatedViTClient(client_id=i, dataset=client_datasets[i], device=device, sparsity_k=0.1)
        for i in range(num_clients)
    ]

    # --- RL Tuner ---
    tuner = RLHyperparamTuner(
        lr_options=[1e-4, 5e-4, 1e-3, 3e-3],
        batch_size_options=[16, 32],
    )

    # --- Baseline: local-only (no federation) ---
    print_banner("Baseline: Local-only training (client 0, round 1 equivalent)")
    local_model = copy.deepcopy(model)
    local_client = FederatedViTClient(client_id=0, dataset=client_datasets[0], device=device)
    local_client.set_model(local_model)
    local_client.train(lr=1e-3, batch_size=32, epochs=local_epochs * num_rounds)
    local_acc = local_client.evaluate(test_dataset)
    print(f"  Local-only accuracy: {local_acc:.4f} ({local_acc*100:.1f}%)")

    # --- Federated training ---
    print_banner("Federated Training with UCB1 AutoRL")
    round_results = []

    for rnd in range(1, num_rounds + 1):
        # UCB1 selects hyperparams
        lr, batch_size, arm_idx = tuner.select()
        print(f"\n  Round {rnd}/{num_rounds} | UCB1 selected: lr={lr:.0e}, batch_size={batch_size}")

        global_model = server.get_global_model()
        all_updates = []
        client_sizes = []

        for client in clients:
            client.set_model(global_model)
            metrics = client.train(lr=lr, batch_size=batch_size, epochs=local_epochs)
            updates = client.get_sparse_updates(global_model)
            all_updates.append(updates)
            client_sizes.append(len(client.dataset))
            print(f"    Client {client.client_id}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")

        # Aggregate (weighted by dataset size)
        server.aggregate(all_updates, client_weights=client_sizes)

        # Evaluate global model
        fed_acc = server.evaluate(test_dataset)
        print(f"  → Federated accuracy: {fed_acc:.4f} ({fed_acc*100:.1f}%)")

        # Update bandit with reward
        tuner.update(arm_idx, reward=fed_acc)
        round_results.append({"round": rnd, "lr": lr, "batch_size": batch_size, "fed_acc": fed_acc})

    # --- Final comparison ---
    print_banner("Results Summary")
    print(f"  Local-only accuracy:      {local_acc:.4f} ({local_acc*100:.1f}%)")
    final_fed_acc = round_results[-1]["fed_acc"]
    print(f"  Federated accuracy:       {final_fed_acc:.4f} ({final_fed_acc*100:.1f}%)")
    delta = final_fed_acc - local_acc
    print(f"  Δ (federated - local):    {delta:+.4f}")

    print("\n  Round-by-round federated accuracy:")
    for r in round_results:
        bar = "█" * int(r["fed_acc"] * 40)
        print(f"    Round {r['round']}: {r['fed_acc']:.4f}  {bar}")

    print()
    print(tuner.summary())

    return {
        "local_acc": local_acc,
        "final_fed_acc": final_fed_acc,
        "rounds": round_results,
        "best_arm": tuner.best_arm(),
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_demo(device=device)
    sys.exit(0)
