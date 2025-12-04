# minute_maker - 必要なライブラリのインストール

## インストール済みライブラリ

uvを使用して以下のライブラリをインストールしました:

### コアライブラリ
- ✅ **pymupdf** (fitz) - PDFファイルの読み込み
- ✅ **pandas** - データフレーム操作
- ✅ **torch** - PyTorch (機械学習フレームワーク)
- ✅ **torchaudio** - 音声処理
- ✅ **python-dotenv** - 環境変数の管理
- ✅ **openai** - Azure OpenAI API
- ✅ **pydub** - 音声ファイルの変換・編集
- ✅ **python-docx** - Word文書の読み込み
- ✅ **python-pptx** - PowerPoint文書の読み込み
- ✅ **extract-msg** - Outlookメッセージファイルの読み込み

### SpeechBrainについて
✅ **speechbrain** - 話者認識用ライブラリ

Python 3.12 と以下のライブラリの組み合わせで正常に動作します：
- `torchaudio==2.5.1`
- `soundfile` (バックエンドとして必須)
- `requests` (依存関係として必須)

プロジェクトの設定ファイル (`.python-version` と `pyproject.toml`) でこれらは適切に設定されています。

## インストールコマンド

```bash
# プロジェクトの初期化とPythonバージョンの固定
uv init --no-workspace
uv python pin 3.12

# 必要なライブラリのインストール
uv sync
```

## 使用方法

```bash
# 仮想環境でPythonスクリプトを実行
uv run python mojiokoshi6.py transcribe audio.mp3 --output result.json

# 要約の生成
uv run python mojiokoshi6.py summary result.json --model gpt-4 --prompt "以下の文字起こしを要約してください"

# 動画から音声を抽出
uv run python mojiokoshi6.py video-to-audio video.mp4 --output audio.mp3
```

## 環境変数

`.env`ファイルに以下の環境変数を設定してください:

```
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_KEY=your_api_key_here
```

## トラブルシューティング

### SpeechBrainのインポートエラー

話者認識機能を使用しない場合は、`mojiokoshi6.py`から以下の行をコメントアウトできます:

```python
# from speechbrain.pretrained import EncoderClassifier
```

そして、`load_speaker_encoder()`および`_compute_embedding_from_wav_bytes()`関数を使用しないようにしてください。
