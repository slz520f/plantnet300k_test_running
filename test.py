import torch
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader


# データセットの定義（例としてCustomDatasetを使用）
test_dataset = CustomDataset(transform=transform)  # transformは必要に応じて定義
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def evaluate_model(weights_path, num_classes, test_loader, use_gpu):
    # ResNet18モデルの初期化
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # 出力層の変更

    # 重みのロード
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

    # 重みを現在のモデルに適用する
    model.load_state_dict(checkpoint['model'], strict=False)

    if use_gpu:
        model = model.to('cuda')

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if use_gpu:
                images, labels = images.to('cuda'), labels.to('cuda')

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# 使用するパラメータを設定
weights_path = 'C:/planet_kadai/plant_github_new/PlantNet-300K/results/xp1/xp1_weights_best_acc.tar'  # ファイル名を追加


num_classes = 17  # クラス数をデータセットに合わせて設定（例: 1081）
test_loader = DataLoader(...)  # あなたのテストデータローダーを設定
use_gpu = torch.cuda.is_available()  # GPUを使用するかどうかのフラグ

# モデルを評価
evaluate_model(weights_path, num_classes, test_loader, use_gpu)
