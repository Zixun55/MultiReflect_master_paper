import torch
from utils import ImageTextToImageModel

# 從 config 載入路徑或直接指定模型 ID
model_id = "OPEA/Llama-3.2V-11B-cot-int4-sym-inc"

try:
    print(f"正在載入模型至 RTX 2080 Ti (11GB)...")
    client = ImageTextToImageModel(model_id)
    print("✅ 環境安裝成功！LLaVA 模型已準備好運作。")
    print(f"目前顯存佔用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
except Exception as e:
    print(f"❌ 載入失敗: {e}")