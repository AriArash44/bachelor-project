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
    parser.add_argument("--in-csv",      required=True, help="Path to preprocessed features CSV")
    parser.add_argument("--out-y-csv",   required=True, help="Output CSV for predictions")
    parser.add_argument("--model-pt",    required=True, help="Trained PyTorch checkpoint (.pt)")
    parser.add_argument("--label-map",   required=True, help="Pickle for label ID → label string")
    parser.add_argument("--input-size",  type=int, default=1, help="Model input size")
    parser.add_argument("--hidden-size", type=int, default=64, help="GRU hidden size")
    parser.add_argument("--num-heads",   type=int, default=4, help="Attention heads")
    parser.add_argument("--num-classes", type=int, default=8, help="Number of output classes")
    args = parser.parse_args()
    print(">> Loading label map...")
    with open(args.label_map, "rb") as f:
        id2label = pickle.load(f)
    print(">> Instantiating MHABiGRU...")
    model = MHABiGRU(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_classes=args.num_classes
    )
    print(">> Loading weights from checkpoint...")
    state = torch.load(args.model_pt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(">> Reading input CSV...")
    df_X = pd.read_csv(args.in_csv)
    X_tensor = torch.tensor(df_X.values, dtype=torch.float32)
    if X_tensor.dim() == 2:
        X_tensor = X_tensor.unsqueeze(1) 
    print(">> Running inference...")
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()
    y_labels = [id2label[i] for i in preds]
    print(f">> Saving predictions to {args.out_y_csv}")
    pd.Series(y_labels, name="y_pred").to_csv(args.out_y_csv, index=False)

if __name__ == "__main__":
    main()
