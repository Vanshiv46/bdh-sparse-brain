"""
export_activations.py
─────────────────────────────────────────────────────────────────────────────
Standalone script to export BDH activation data from a trained checkpoint.
Works with the output of train.py from pathwaycom/bdh.

USAGE (from inside the bdh/ repo directory):
    python export_activations.py
    python export_activations.py --checkpoint out/ckpt.pt
    python export_activations.py --checkpoint out/ckpt.pt --tokens "London Paris Dollar"
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import json
import argparse
import os


def get_char_encoder():
    """Tiny shakespeare / BDH default char-level encoder."""
    chars = (
        "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
    )
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def load_bdh_model(checkpoint_path, device="cpu"):
    """Load a BDH model from a train.py checkpoint."""
    try:
        from bdh import BDH, BDHConfig
    except ImportError:
        raise RuntimeError(
            "Cannot import BDH. Run this script from inside the pathwaycom/bdh directory.\n"
            "  cd /path/to/bdh\n"
            "  python /path/to/export_activations.py"
        )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        config_args = ckpt.get("model_args", {})
        config = BDHConfig(**config_args)
        model = BDH(config)
        # Strip DDP prefix if present
        state_dict = ckpt["model"]
        unwanted = "_orig_mod."
        state_dict = {
            (k[len(unwanted):] if k.startswith(unwanted) else k): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
        print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        print("No checkpoint — building default small BDH model with random weights.")
        config = BDHConfig(vocab_size=95, block_size=64, n_layer=4, n_head=4, n_embd=128)
        model = BDH(config)

    model.to(device).eval()
    return model


def extract_layer_activations(model, token_ids, device="cpu"):
    """
    Run a forward pass and collect ReLU activation statistics at each layer.
    Returns list of dicts: [{layer, sparsity, active_count, total, indices}, ...]
    """
    records = []

    def hook_fn(name):
        def fn(module, inp, out):
            with torch.no_grad():
                fired   = (out > 0).float()
                total   = int(fired.numel())
                active  = int(fired.sum().item())
                sparse  = round(active / total, 4) if total > 0 else 0
                # Top active neuron indices (capped for JSON size)
                flat = fired.view(-1)
                indices = flat.nonzero(as_tuple=False).view(-1).tolist()[:300]
                records.append({
                    "layer":   name,
                    "sparsity": sparse,
                    "active_count": active,
                    "total_neurons": total,
                    "active_indices": indices,
                })
        return fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return records


def process_tokens(model, token_list, stoi, device):
    """Run inference for each token/phrase and collect sparsity data."""
    results = []
    for text in token_list:
        ids = [stoi.get(c, 0) for c in text]
        if not ids:
            continue
        records = extract_layer_activations(model, ids, device)
        if not records:
            continue
        avg_sparsity = sum(r["sparsity"] for r in records) / len(records)
        per_layer = {}
        for r in records:
            per_layer.setdefault(r["layer"], []).append(r["sparsity"])
        per_layer_avg = {k: round(sum(v)/len(v), 4) for k, v in per_layer.items()}

        results.append({
            "text": text,
            "overall_sparsity": round(avg_sparsity, 4),
            "per_layer": per_layer_avg,
            "raw_records": records[:6],
            "token_count": len(ids),
        })
        print(f"  '{text}' → {avg_sparsity:.1%} active neurons")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to BDH checkpoint (e.g. out/ckpt.pt)")
    parser.add_argument("--output", type=str, default="activations_output.json",
                        help="Output JSON path")
    parser.add_argument("--tokens", type=str, default=None,
                        help="Space-separated tokens/phrases to test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    DEFAULT_TOKENS = [
        "London", "Paris", "Berlin", "Tokyo",
        "dollar", "euro", "pound", "yen",
        "France", "Germany", "England",
        "the cat sat on the mat",
        "neural networks learn from data",
        "is are the a",
    ]

    tokens = args.tokens.split() if args.tokens else DEFAULT_TOKENS

    print("\n🐉 BDH Activation Exporter")
    print("=" * 50)

    model = load_bdh_model(args.checkpoint, device)
    stoi, _ = get_char_encoder()

    print(f"\nProcessing {len(tokens)} tokens/phrases...")
    results = process_tokens(model, tokens, stoi, device)

    # Build export dict
    output = {
        "meta": {
            "model": "BDH",
            "checkpoint": args.checkpoint or "random_weights",
            "n_tokens_tested": len(results),
            "device": device,
        },
        "transformer_baseline": {
            "activation_density": 0.94,
            "note": "SoftMax ensures near-uniform activation. Source: BDH paper §6.4"
        },
        "sentences": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    avg = sum(r["overall_sparsity"] for r in results) / len(results) if results else 0
    print(f"\n✅ Saved to: {args.output}")
    print(f"\n📊 Results Summary:")
    print(f"   BDH average sparsity:   {avg:.1%}")
    print(f"   Transformer density:    94.0%")
    print(f"   Ratio:                  {0.94/avg:.1f}× more efficient")
    print(f"\n→ Copy {args.output} to bdh-sparse-brain/assets/activations.json")


if __name__ == "__main__":
    main()
