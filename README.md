# 前眼部画像からの年齢推定モデル

参考論文: Guo et al., Br J Ophthalmol 2024 "Crystalline lens nuclear age prediction as a new biomarker of nucleus degeneration"

## プロジェクト構成

```
age_estimation/
├── config.py           # 設定クラス
├── dataset.py          # データローダー
├── models.py           # 5モデル定義
├── trainer.py          # 学習ループ、LR Range Test、Early Stopping
├── evaluation.py       # 評価メトリクス（MAE, RMSE, r + Bootstrap CI）
├── utils.py            # 結果保存（CSV/JSON）
├── main_phase1.py      # Phase 1 メインスクリプト
├── requirements.txt    # 依存パッケージ
└── README.md           # このファイル
```

## セットアップ

```bash
pip install -r requirements.txt
```

## データフォーマット

CSVファイルは以下のカラムを含む必要があります:
- `image_path` (または `path`, `file`, `image`を含む列名): 画像ファイルへのパス
- `age` (または `label`, `year`を含む列名): 年齢ラベル

例:
```csv
image_path,age
/path/to/image1.jpg,45.0
/path/to/image2.jpg,62.0
```

## Phase 1: 5モデル比較

### 基本的な使用法

```bash
python main_phase1.py \
    --train_csv /path/to/train.csv \
    --test_csv /path/to/test.csv \
    --output_dir ./results/phase1
```

### すべてのオプション

```bash
python main_phase1.py \
    --train_csv /path/to/train.csv \
    --test_csv /path/to/test.csv \
    --output_dir ./results/phase1 \
    --models resnet50 efficientnet_b7 vit_base_patch16_224 \
    --batch_size 16 \
    --n_folds 5 \
    --max_epochs 100 \
    --patience 5 \
    --seed 42 \
    --num_workers 4
```

### 特定のモデルのみ実行

```bash
python main_phase1.py \
    --train_csv /path/to/train.csv \
    --test_csv /path/to/test.csv \
    --models resnet50 vit_base_patch16_224
```

## 出力

```
results/phase1/
├── cv_summary.csv              # CV結果サマリー
├── test_results.csv            # テストセット結果
├── experiment_log.json         # 実験ログ
├── {model}_cv_results.json     # 各モデルのCV詳細結果
├── {model}_test_predictions.csv # 各モデルの予測値
└── {model}_fold{n}.pth         # 学習済みモデル
```

## 比較モデル

| モデル名 | timm名 |
|---------|--------|
| ResNet-50 | resnet50 |
| EfficientNet-B7 | efficientnet_b7 |
| ViT-Base | vit_base_patch16_224 |
| ViT-Large | vit_large_patch16_224 |
| Swin-Base | swin_base_patch4_window7_224 |

## 学習設定

- **事前学習**: ImageNet-1k
- **オプティマイザ**: AdamW
- **学習率**: LR Range Testで決定（候補: 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3）
- **バッチサイズ**: 16
- **正規化**: ImageNet mean/std
- **データ拡張**: なし
- **Early Stopping**: patience=5
- **損失関数**: MSE Loss

## 評価指標

- **主指標**: MAE (Mean Absolute Error)
- **参考指標**: RMSE, Pearson r
- **信頼区間**: Bootstrap法（95% CI, 1000回反復）
# AnteriorEyeAgeDetection
