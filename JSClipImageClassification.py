# # Japanese Stable CLIP による画像分類

# ## 環境設定

# ### ライブラリーのインストール（Python 仮想環境の利用推奨）

# ### ライブラリーのインポート

from typing import Union, List
import ftfy, html, re
from pathlib import Path
import torch
from huggingface_hub import notebook_login
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, BatchFeature
import gradio as gr
import os

# ### デモ用サンプル画像のダウンロード

if not os.path.exists('samples'):
    os.makedirs('samples')

sample_images = [
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hauskatze_langhaar.jpg/800px-Hauskatze_langhaar.jpg?20110412192053', 'samples/sample01.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Black_barn_cat_-_Public_Domain_%282014_photo%3B_cropped_2022%29.jpg/800px-Black_barn_cat_-_Public_Domain_%282014_photo%3B_cropped_2022%29.jpg?20220510154737', 'samples/sample02.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Airbus_A380.jpg/640px-Airbus_A380.jpg', 'samples/sample03.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/DeltaIVHeavy_NROL82_launch.jpg/640px-DeltaIVHeavy_NROL82_launch.jpg', 'samples/sample04.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/011_The_lion_king_Tryggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg/640px-011_The_lion_king_Tryggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg', 'samples/sample05.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Prunus_incisa_var._kinkiensis_%27Kumagaizakura%27_07.jpg/640px-Prunus_incisa_var._kinkiensis_%27Kumagaizakura%27_07.jpg', 'samples/sample06.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Camila_Moreno_2.jpg/640px-Camila_Moreno_2.jpg', 'samples/sample07.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/USSRC_Rocket_Park.JPG/800px-USSRC_Rocket_Park.JPG', 'samples/sample08.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/f/fc/Kintamani_dog_white.jpg', 'samples/sample09.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Man_biking_on_Recife_city.jpg/800px-Man_biking_on_Recife_city.jpg', 'samples/sample10.png')
]

for url, filename in sample_images:
    if not os.path.exists(filename):
        os.system(f'wget {url} -O {filename}')


# ## Japanese Stable CLIP モデルのロード

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "stabilityai/japanese-stable-clip-vit-l-16"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

# %% [markdown]
# ## トークン化関数の定義

# %% [markdown]
# ### トークン化関数で使用するヘルパー関数の定義
# https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py#L65C8-L65C8 から引用

# %%
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# %% [markdown]
# ### トークン化関数の定義（分類のターゲットとなるカテゴリを表すテキストをトークン化する関数）
# CLIPのオリジナルコードから引用（https://github.com/openai/CLIP/blob/main/clip/clip.py#L195）

# %%
def tokenize(
    texts: Union[str, List[str]],
    max_seq_len: int = 77,
):
    if isinstance(texts, str):
        texts = [texts]
    texts = [whitespace_clean(basic_clean(text)) for text in texts]

    inputs = tokenizer(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[tokenizer.bos_token_id] + ids for ids in inputs["input_ids"]]
    attention_mask = [[1] + am for am in inputs["attention_mask"]]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    return BatchFeature(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }
    )




# %% [markdown]
# ## エンベディングを生成する関数を定義
# Japanese Stable CLIP を使ってテキストと画像のエンベディングを生成する関数
# 

# %%
def compute_text_embeddings(text):
  if isinstance(text, str):
    text = [text]
  text = tokenize(texts=text)
  text_features = model.get_text_features(**text.to(device))
  text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
  del text
  return text_features.cpu().detach()

def compute_image_embeddings(image):
  image = processor(images=image, return_tensors="pt").to(device)
  with torch.no_grad():
    image_features = model.get_image_features(**image)
  image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
  del image
  return image_features.cpu().detach()

# %% [markdown]
# ## デモ画面の準備

# %% [markdown]
# ### 画像を分類するカテゴリー（起動時デフォルト）を定義

# %%
categories = [
    "配達員",
    "営業",
    "消防士",
    "救急隊員",
    "自衛隊",
    "スポーツ選手",
    "警察官",
    "動物",
    "猫",
    "犬",
    "うさぎ",
    "ライオン",
    "虎",
    "乗り物",
    "飛行機",
    "自動車",
    "自転車",
    "タクシー",
    "バイク",
    "ロケット",
    "宇宙船",
    "兵器",
    "戦闘機",
    "戦車",
    "空母",
    "護衛艦",
    "潜水艦",
    "ミサイル",
    "植物",
    "鳥",
    "魚",
    "花",
    "樹木",
    "人間",
    "子供",
    "老人",
    "女性",
    "男性",
    "少女",
    "少年",
    "サッカー選手",
    "アスリート",
    "アニメキャラクター",
    "ぬいぐるみ",
    "妖精",
    "エルフ",
    "天使",
    "魔性使い",
    "魔法少女",
    "歌姫",
    "歌手"
]

# %% [markdown]
# ### カテゴリを表すテキストのエンベディングを生成

# %%
text_embeds = compute_text_embeddings(categories)

# %% [markdown]
# ## デモの実行

# %%

TOP_K = 3 # 上位3つのカテゴリを表示

def update_categories_fn(new_categories, state): # カテゴリーを更新ボタンが押された時に実行される関数
    categories = [cat.strip() for cat in new_categories.split(',')]
    text_embeds = compute_text_embeddings(categories)
    state['categories'] = categories
    state['text_embeds'] = text_embeds
    return ', '.join(categories), state

def update_categories_default_fn(state):
    text_embeds = compute_text_embeddings(categories)
    state['categories'] = categories
    state['text_embeds'] = text_embeds
    return ', '.join(categories), state

def inference_fn(img, state): # 分類実行ボタンが押された時に実行される関数
  if img is None:
    return "分類する画像をアップロードするか Examples から選択してください"
  text_embeds = state['text_embeds']
  categories = state['categories']
  num_categories = len(categories)
  image_embeds = compute_image_embeddings(img) # 画像のエンベディングを生成
  similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1) # 画像のエンベディングとテキストのエンベディングの内積を計算
  similarity = similarity[0].numpy().tolist() # 内積の結果をリストに変換
  output_dict = {categories[i]: float(similarity[i]) for i in range(num_categories)} # カテゴリと確信度の辞書を生成
  del image_embeds # 画像のエンベディングを削除
  return output_dict # カテゴリと確信度の辞書を返す


with gr.Blocks(title="Japanese Stable CLIP") as demo: # Gradio による画面の作成
    state = gr.State({
            "categories": categories,
            "text_embeds": text_embeds
        })
    gr.Markdown("# Japanese Stable CLIP による画像の分類（a.k.a. 画像によるテキストの検索）")
    with gr.Row():
      with gr.Column():
        inp = gr.Image(label="分類したい画像", type="pil", height=512)
        btn = gr.Button("分類実行")
        
      with gr.Column():
        out = gr.Label(label="分類結果", num_top_classes=TOP_K)
        with gr.Accordion("分類先のカテゴリ一覧", open=False):
            categories_input = gr.Textbox(label="カテゴリー（カンマ区切り）", value=", ".join(categories), lines=4)
            with gr.Row():
              with gr.Column():
                category_update_btn = gr.Button("カテゴリーを更新")
                category_update_btn.click(fn=update_categories_fn, inputs=[categories_input, state], outputs=[categories_input, state])
              with gr.Column():
                category_update_default_btn = gr.Button("デフォルトに戻す")
                category_update_default_btn.click(fn=update_categories_default_fn, inputs=[state], outputs=[categories_input, state])
        with gr.Accordion("Japanese Stable CLIP について", open=False):
          with gr.Column():
            gr.Markdown(
            """[Japanese Stable CLIP](https://huggingface.co/stabilityai/japanese-stable-clip-vit-l-16) is a [CLIP](https://arxiv.org/abs/2103.00020) model by [Stability AI](https://ja.stability.ai/).
                  - Blog: https://ja.stability.ai/blog/japanese-stable-clip
                  - Twitter: https://twitter.com/StabilityAI_JP
                  - Discord: https://discord.com/invite/StableJP"""
            )
    with gr.Row():
      examples = gr.Examples(
        examples=[str(p) for p in Path("samples").glob("*.png")],
        inputs=inp
      )
    btn.click(fn=inference_fn, inputs=[inp, state], outputs=[out])

if __name__ == "__main__":
    demo.launch(debug=True, share=True, inbrowser=True)

# %%



