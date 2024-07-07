import torch
from transformers import AutoModel, AutoTokenizer

# モデルとトークナイザーのロード
model_name = "stabilityai/japanese-stable-clip-vit-l-16"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# テキストのエンベディングを取得
text = "自動車"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model.get_text_features(**inputs)

# エンベディングのL2ノルムを計算
embedding = outputs[0]
norm = torch.norm(embedding, p=2).item()

# ノルムが1に近いか確認
is_normalized = torch.isclose(torch.tensor(norm), torch.tensor(1.0))
print(f"エンベディングは正規化されていますか？ {is_normalized}")