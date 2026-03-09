# AutoSign (Forked Version)
本リポジトリは、連続手話認識モデル[AutoSign](https://github.com/EbimoJohnny/AutoSign)をフォークしたものです。  
本リポジトリはオリジナル版と同様にMITライセンスのもとで公開されています。  
オリジナルの研究コードや詳細についてはフォーク元を参照してください。

## 概要
AutoSignは、手話の動画からキーポイントを抽出し、事前学習されたTransformerベースの言語モデルを用いて直接テキストへの翻訳を行うアプローチです。  
本リポジトリでは事前学習済みの言語モデルを用いず、ランダムな初期重みから手話翻訳モデルをゼロから学習させる構成をデフォルトとしています。  
これにより、任意のデータセットで学習が行えるようになっています。

## データセット
学習・評価用のデータセットについては、[フォーク元のオリジナルリポジトリ](https://github.com/EbimoJohnny/AutoSign)の `Dataset` セクション、または `annotation_v2` ディレクトリを参照してください。  
顔のキーポイントを除外した右手・左手・上半身の計67点のキーポイントを使用しています。  
また、学習にはz座標を用いておらず、x座標とy座標の2次元特徴のみを入力としています。

## インストール手順
本リポジトリでは `uv` を使用したパッケージ管理を行っています。
```bash
git clone git@github.com:Tamagosushio/AutoSign.git
cd AutoSign
uv sync
```

## 学習の実行
モデルをゼロから学習する場合は、以下のコマンドを実行します。
```bash
uv run main.py --mode custom --gpu 0 --num_runs 10 --epochs 100 --batch_size 64
```
- `--mode`: 使用するデータや保存ディレクトリの識別
- `--gpu`: 使用するGPUのID（デフォルト: 0）
- `--num_runs`: 学習プロセスの実行回数（デフォルト: 1）
- `--epochs`: 学習の総エポック数（デフォルト: 100）
- `--batch_size`: バッチサイズ（デフォルト: 64）
- `--disable_augmentation`: データ拡張（フレームの欠落、回転、ノイズなど）を無効化する

## Citation (Original Work)
If you use the original AutoSign foundation in your research, please cite:
```bibtex
@article{johnny2025autosign,
  title={AutoSign: Direct Pose-to-Text Translation for Continuous Sign Language Recognition},
  author={Johnny, Samuel Ebimobowei and Guda, Blessed and Stephen, Andrew Blayama and Gueye, Assane},
  journal={arXiv preprint arXiv:2507.19840},
  year={2025}
}
```

## Acknowledgments
- **DTrOCR**: We acknowledge the decoder-only transformer architecture principles from [DTrOCR: Decoder-only Transformer for Optical Character Recognition](https://doi.org/10.48550/arXiv.2308.15996).
- **AraGPT2**: Our implementation leverages the Arabic GPT-2 model developed by [AUB-MIND](https://github.com/aub-mind/arabert).
- **ICCV MSLR 2025**: Thanks to the organizers of the [ICCV 2025 Multilingual Sign Language Recognition Challenge](https://iccv-mslr-2025.github.io/MSLR/).