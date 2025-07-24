import torch
import torch.nn as nn
import argparse
import pickle
import pandas as pd

class MHABiGRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_heads: int,
                 num_classes: int,
                 dropout: float = 0.2):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.bigru(x)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        pooled = attn_out.mean(dim=1)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--out-y-csv", required=True)
    parser.add_argument("--model-pt", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--input-size", type=int, default=11)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()
    print(">> Loading label map...")
    with open(args.label_map, "rb") as f:
        label_encoder = pickle.load(f)
    trained_classes = len(label_encoder.classes_)
    id2label = {i: label_encoder.classes_[i] for i in range(trained_classes)}
    print(f">> Trained classes: {trained_classes}, Desired output classes: {args.num_classes}")
    model = MHABiGRU(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_classes=trained_classes
    )
    print(">> Loading weights from checkpoint...")
    state = torch.load(args.model_pt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    if trained_classes < args.num_classes:
        print(f">> Expanding output layer to {args.num_classes}...")
        old_fc = model.fc
        new_fc = nn.Linear(old_fc.in_features, args.num_classes)
        with torch.no_grad():
            new_fc.weight[:trained_classes] = old_fc.weight
            new_fc.bias[:trained_classes] = old_fc.bias
        model.fc = new_fc
    model.eval()
    print(">> Reading input CSV...")
    df_X = pd.read_csv(args.in_csv)
    X_tensor = torch.tensor(df_X.values, dtype=torch.float32)
    if X_tensor.dim() == 2:
        X_tensor = X_tensor.unsqueeze(1)
    print(">> Running inference (10-class probability output)...")
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    col_names = [id2label.get(i, f"unused_{i}") for i in range(args.num_classes)]
    df_probs = pd.DataFrame(probs, columns=col_names)
    print(f">> Saving probability vectors to {args.out_y_csv}")
    df_probs.to_csv(args.out_y_csv, index=False)

if __name__ == "__main__":
    main()