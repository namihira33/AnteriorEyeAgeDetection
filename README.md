# 年齢推定モデル分析ツール

論文 Guo et al., Br J Ophthalmol 2024 "Crystalline lens nuclear age prediction" に基づく前眼部画像からの年齢推定モデルの分析ツールです。

## 新規追加ファイル

### 0. `trainer_updated.py` & `main_phase1_updated.py` - 最終モデル訓練機能

**重要な追加機能**: CVの結果を使って全訓練データで最終モデルを訓練する機能を追加

既存のコードでは各foldのモデルのみ保存されていましたが、以下のワークフローが可能に：
1. 5分割交差検証でハイパーパラメータ（学習率、エポック数）を決定
2. **全訓練データで最終モデルを訓練**（`--train_final` オプション）
3. 最終モデルまたはアンサンブルでテスト評価

### 1. `visualization.py` - 注目領域可視化モジュール

以下の可視化手法を実装：
- **Grad-CAM**: 勾配ベースのClass Activation Mapping
- **Grad-CAM++**: 改良版Grad-CAM（より正確な注目領域）
- **Score-CAM**: 勾配不要のCAM（より安定した結果）
- **Attention Rollout**: Vision Transformer用の注目領域可視化

### 2. `analysis.py` - 分析用メインスクリプト

以下の分析機能を提供：
- モデル間の統計的比較（paired t-test, Wilcoxon検定）
- 年齢層別のMAE分析と信頼区間
- 出版用図表の自動生成
- 詳細な分析レポート（Markdown形式）

### 3. `evaluation_updated.py` - 評価モジュール（更新版）

既存の `evaluation.py` を以下の機能で拡張：
- 詳細な統計量の計算
- 年齢層別メトリクス（Bootstrap CI付き）
- モデル間のペア比較（効果量Cohen's d含む）
- LaTeX形式の表生成

## セットアップ

```bash
# 新しいファイルをプロジェクトに追加
cp visualization.py analysis.py /path/to/your/project/
cp evaluation_updated.py /path/to/your/project/evaluation.py  # 既存ファイルを置き換え

# 追加の依存パッケージをインストール
pip install opencv-python seaborn
```

## 実行例

### Phase 1: モデル比較（最終モデル訓練あり）

```bash
# trainer.py と main_phase1.py を更新版に置き換え
cp trainer_updated.py trainer.py
cp main_phase1_updated.py main_phase1.py

# CV + 最終モデル訓練
python main_phase1.py \
    --train_csv ./csv/train.csv \
    --test_csv ./csv/test.csv \
    --train_image_dir ./images/normal20251210/training \
    --test_image_dir ./images/normal20251210/test \
    --output_dir ./results/phase1 \
    --train_final

# 最終モデルでテスト評価（アンサンブルではなく単一モデル）
python main_phase1.py \
    --train_csv ./csv/train.csv \
    --test_csv ./csv/test.csv \
    --train_image_dir ./images/normal20251210/training \
    --test_image_dir ./images/normal20251210/test \
    --output_dir ./results/phase1 \
    --train_final \
    --use_final_for_test
```

### 出力されるモデルファイル

```
results/phase1/
├── resnet50_fold1.pth          # Fold 1 モデル
├── resnet50_fold2.pth          # Fold 2 モデル
├── ...
├── resnet50_fold5.pth          # Fold 5 モデル
├── resnet50_final.pth          # ★全訓練データで訓練した最終モデル
├── resnet50_cv_results.json    # CV結果（最適LR、エポック数含む）
├── resnet50_final_results.json # 最終モデル訓練結果
└── ...
```

### 基本的な分析の実行

```bash
python analysis.py \
    --models_dir ./results/phase1 \
    --test_csv ./csv/test.csv \
    --image_dir ./images/normal20251210/test \
    --output_dir ./analysis_results \
    --models resnet50 efficientnet_b7 vit_base_patch16_224
```

### すべてのモデルを分析

```bash
python analysis.py \
    --models_dir ./results/phase1 \
    --test_csv ./csv/test.csv \
    --image_dir ./images/normal20251210/test \
    --output_dir ./analysis_results \
    --models resnet50 efficientnet_b7 vit_base_patch16_224 vit_large_patch16_224 swin_base_patch4_window7_224
```

### CAM可視化のみスキップ（高速化）

```bash
python analysis.py \
    --models_dir ./results/phase1 \
    --test_csv ./csv/test.csv \
    --image_dir ./images/normal20251210/test \
    --output_dir ./analysis_results \
    --skip_cam
```

### CAMサンプル数を指定

```bash
python analysis.py \
    --models_dir ./results/phase1 \
    --test_csv ./csv/test.csv \
    --image_dir ./images/normal20251210/test \
    --output_dir ./analysis_results \
    --cam_samples 50
```

## 出力ファイル

```
analysis_results/
├── analysis_report.md              # 分析レポート（Markdown）
├── statistical_comparison.csv      # モデル間統計比較
├── figures/
│   ├── scatter_comparison.png      # 散布図比較（出版用）
│   ├── scatter_comparison.pdf      # 散布図比較（PDF版）
│   ├── model_comparison_bar.png    # MAE棒グラフ（95% CI付き）
│   └── model_comparison_bar.pdf    # MAE棒グラフ（PDF版）
├── {model_name}/
│   ├── error_distribution.png      # 誤差分布図
│   ├── error_by_age.csv            # 年齢層別MAE
│   ├── error_by_age.png            # 年齢層別MAE図
│   ├── cams/
│   │   ├── average_cam.png         # 平均CAM
│   │   └── ...
│   └── age_group_analysis/
│       └── age_group_comparison.png # 年齢層別注目領域
└── ...
```

## Pythonスクリプトからの使用例

### Grad-CAMの生成

```python
from visualization import GradCAM, get_target_layer, visualize_cam
from models import AgeEstimationModel
from dataset import get_transforms
from PIL import Image
import torch

# モデル読み込み
model = AgeEstimationModel('resnet50', pretrained=False)
model.load_state_dict(torch.load('results/phase1/resnet50_fold1.pth'))
model.eval()

# ターゲット層を取得
target_layer = get_target_layer(model, 'resnet50')

# Grad-CAM生成器
gradcam = GradCAM(model, target_layer)

# 画像前処理
transform = get_transforms(image_size=224, is_train=False)
image = Image.open('test_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# CAM生成
cam = gradcam(input_tensor)

# 可視化
import numpy as np
import matplotlib.pyplot as plt

original = np.array(image.resize((224, 224))) / 255.0
overlay = visualize_cam(original, cam)

plt.imshow(overlay)
plt.axis('off')
plt.savefig('gradcam_result.png')
```

### Bootstrap信頼区間の計算

```python
from evaluation_updated import bootstrap_all_metrics, format_metric_with_ci
import numpy as np

# 予測値と正解値
predictions = np.array([45.2, 52.1, 38.5, ...])
targets = np.array([47.0, 50.0, 40.0, ...])

# メトリクス計算（95% CI付き）
metrics = bootstrap_all_metrics(
    predictions, targets,
    n_iterations=1000,
    ci=0.95
)

# 結果表示
print(f"MAE: {format_metric_with_ci(metrics['mae'])}")
print(f"RMSE: {format_metric_with_ci(metrics['rmse'])}")
print(f"Pearson r: {format_metric_with_ci(metrics['pearson_r'])}")
```

### モデル間の統計比較

```python
from evaluation_updated import compare_predictions_paired

# 2つのモデルの予測を比較
comparison = compare_predictions_paired(
    predictions_model1,
    predictions_model2,
    targets,
    n_bootstrap=1000
)

print(f"MAE差: {comparison['mean_difference']:.3f}")
print(f"95% CI: [{comparison['diff_ci_lower']:.3f}, {comparison['diff_ci_upper']:.3f}]")
print(f"Wilcoxon p-value: {comparison['wilcoxon_pvalue']:.4f}")
print(f"Cohen's d: {comparison['cohens_d']:.3f}")
```

### 年齢層別分析

```python
from evaluation_updated import calculate_age_stratified_metrics

# 年齢層別メトリクス
age_results = calculate_age_stratified_metrics(
    predictions, targets,
    age_bins=[0, 30, 40, 50, 60, 70, 100],
    n_bootstrap=1000
)

for age_group, metrics in age_results.items():
    print(f"{age_group}歳: MAE={metrics['mae']:.2f} "
          f"(95% CI: {metrics['mae_ci_lower']:.2f}-{metrics['mae_ci_upper']:.2f}), "
          f"n={metrics['n']}")
```

## 注意事項

### 訓練フローについて

**推奨ワークフロー**:

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 5分割交差検証 (CV)                              │
│   - 各foldで LR Range Test → 最適学習率を決定           │
│   - 各foldで訓練 → Early Stopping で最適エポック数決定  │
│   - 各fold モデルを保存 ({model}_fold1~5.pth)           │
│   - CV MAE を計算（モデル選択の指標）                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: 最終モデル訓練 (--train_final)                  │
│   - CVで決定した最適LR（中央値）を使用                  │
│   - CVで決定した最適エポック数（中央値）を使用          │
│   - 全訓練データで訓練                                  │
│   - 最終モデルを保存 ({model}_final.pth)                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: テスト評価                                      │
│   オプション A: アンサンブル（デフォルト）               │
│     - fold1~5の予測値を平均                             │
│   オプション B: 最終モデル (--use_final_for_test)       │
│     - 全データで訓練した単一モデルで予測                │
└─────────────────────────────────────────────────────────┘
```

**アンサンブル vs 最終モデル**:
- アンサンブル: 一般的に性能が安定、論文でよく使用される
- 最終モデル: デプロイ時にシンプル、推論時間が短い

### その他の注意事項

1. **Grad-CAMのターゲット層**: モデルアーキテクチャによって適切なターゲット層が異なります。`get_target_layer()` 関数で自動的に選択されますが、カスタムモデルの場合は調整が必要な場合があります。

2. **メモリ使用量**: Score-CAMは大量のメモリを使用します。GPUメモリが不足する場合は Grad-CAM または Grad-CAM++ を使用してください。

3. **Vision Transformer**: ViT系モデルではGrad-CAMの結果が不安定な場合があります。Attention Rolloutも試してみてください。

4. **Bootstrap反復回数**: デフォルトは1000回です。より正確なCIが必要な場合は増やしてください（計算時間が増加します）。

## 参考文献

- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
- Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks", WACV 2018
- Wang et al., "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks", CVPR 2020
- Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020
- Guo et al., "Crystalline lens nuclear age prediction as a new biomarker of nucleus degeneration", Br J Ophthalmol 2024
