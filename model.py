import torch
from torchvision.models import resnet18
from utils import load_model  # load_model関数が適切に実装されている前提

# モデルの定義
num_classes = 17  # Pl@ntNet-300Kのクラス数
model = resnet18(num_classes=num_classes)

# GPU使用の設定
use_gpu = True
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# 重みファイルのパス
filename = 'C:\Users\1234\Downloads\resnet18_weights_best_acc.tar'

# モデルの重みをロード
try:
    load_model(model, filename=filename, use_gpu=use_gpu)
except Exception as e:
    print(f"Error loading model: {e}")

# 推論のための準備（例）
model.eval()  # モデルを評価モードに設定

# ここからは、データローダーを使用して推論を行うことができます
# 例:
# test_loader = DataLoader(...)

# for images, labels in test_loader:
#     images = images.to(device)
#     outputs = model(images)
#     # 出力処理を行う
