# minute_maker - インストールガイド

## 前提条件

- **Python**: 3.12 以降
- **Node.js**: 18 以降
- **uv**: Python パッケージマネージャー ([インストール方法](https://github.com/astral-sh/uv))

## インストール手順

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd minute_maker
```

### 2. バックエンドのセットアップ

#### 2.1 Python環境の初期化

```bash
# プロジェクトの初期化とPythonバージョンの固定
uv init --no-workspace
uv python pin 3.12

# 必要なライブラリのインストール
uv sync
```

#### 2.2 環境変数の設定

`.env`ファイルをプロジェクトルートに作成し、以下の環境変数を設定してください:

**Azure OpenAI使用時:**
```env
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_KEY=your_api_key_here
```

**OSS版Whisper（faster-whisper）使用時:**
```env
WHISPER_PROVIDER=faster-whisper
OSS_WHISPER_MODEL=base  # tiny, base, small, medium, large-v2, large-v3
OSS_WHISPER_DEVICE=cpu  # cpu または cuda（GPU使用時）
```

**注意**: OSS版Whisperを使用する場合、Azure OpenAI設定は不要です。

#### 2.3 SpeechBrain モデルのダウンロード（話者識別機能を使う場合）

**重要**: 話者認識機能を使用するには、事前にSpeechBrainモデルをダウンロードする必要があります。

```bash
# モデルダウンロードスクリプトを実行
uv run python download_model.py
```

このコマンドは以下を実行します:
- Hugging Faceから`speechbrain/spkrec-ecapa-voxceleb`モデルをダウンロード
- `backend/tmp_model/`ディレクトリに保存（約89.1 MB）
- 必要なファイル:
  - `embedding_model.ckpt` (79.46 MB)
  - `classifier.ckpt` (5.28 MB)
  - `label_encoder.txt` (0.12 MB)
  - その他の設定ファイル

**注意**: ダウンロードには数分かかる場合があります。インターネット接続が必要です。

#### 2.4 バックエンドサーバーの起動

```bash
uv run uvicorn backend.app.main:app --reload
```

サーバーは `http://localhost:8000` で起動します。

### 3. フロントエンドのセットアップ

#### 3.1 依存関係のインストール

```bash
cd frontend
npm install
```

#### 3.2 開発サーバーの起動

```bash
npm run dev -- --host
```

フロントエンドは `http://localhost:5173` で起動します。

## インストール済みライブラリ

### バックエンド（Python）

#### コアライブラリ
- ✅ **pymupdf** (fitz) - PDFファイルの読み込み
- ✅ **pandas** - データフレーム操作
- ✅ **torch** - PyTorch (機械学習フレームワーク)
- ✅ **torchaudio** - 音声処理
- ✅ **python-dotenv** - 環境変数の管理
#### 文字起こしライブラリ
- ✅ **openai** - Azure OpenAI API（オプション、Azure使用時のみ必要）
- ✅ **faster-whisper** - OSS版Whisper（オプション、ローカル文字起こし用）

#### 音声処理
- ✅ **pydub** - 音声ファイルの変換・編集
- ✅ **python-docx** - Word文書の読み込み
- ✅ **python-pptx** - PowerPoint文書の読み込み
- ✅ **extract-msg** - Outlookメッセージファイルの読み込み

#### 話者認識
- ✅ **speechbrain** - 話者認識用ライブラリ
- ✅ **huggingface-hub** - モデルダウンロード用

Python 3.12 と以下のライブラリの組み合わせで正常に動作します:
- `torchaudio==2.5.1`
- `soundfile` (バックエンドとして必須)
- `requests` (依存関係として必須)

### フロントエンド（Node.js）
- React 18
- TypeScript
- Vite
- Lucide React (アイコン)

## API エンドポイント

バックエンドサーバー起動後、以下のエンドポイントが利用可能です:

- **ヘルスチェック**: `GET /`
- **議事録一覧・登録**: `GET/POST /api/minutes`
- **議事録詳細**: `GET /api/minutes/{id}`
- **音声処理**: `POST /api/process_audio`
- **話者登録**: `POST /api/register_speaker`
- **話者一覧**: `GET /api/speakers`
- **話者追加**: `POST /api/speakers`
- **話者削除**: `DELETE /api/speakers/{name}`
- **話者埋め込み生成**: `POST /api/create_speaker_embedding`
- **要約プロンプト一覧**: `GET /api/prompts`
- **要約生成**: `POST /api/generate_summary`
- **議事録ダウンロード**: `GET /api/minutes/{id}/download?format=docx|xlsx`

Swagger UIドキュメント: `http://localhost:8000/docs`

## トラブルシューティング

### SpeechBrainモデルのダウンロードに失敗する場合

1. **インターネット接続を確認**
   ```bash
   ping huggingface.co
   ```

2. **huggingface-hubを最新版にアップデート**
   ```bash
   uv add --upgrade huggingface-hub
   ```

3. **プロキシ設定を確認**
   ```bash
   # プロキシ環境変数を設定
   export HTTP_PROXY=http://proxy.example.com:8080
   export HTTPS_PROXY=http://proxy.example.com:8080
   ```

4. **手動でモデルファイルをダウンロード**
   - [Hugging Face](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)からファイルをブラウザでダウンロード
   - `backend/tmp_model/`ディレクトリに配置

### torchaudioのインポートエラー

torchaudioのバージョンが `2.5.1` であることを確認してください:

```bash
uv run python -c "import torchaudio; print(torchaudio.__version__)"
```

### 話者認識機能を使用しない場合

話者認識機能を使わない場合は、モデルダウンロード（ステップ 2.3）をスキップできます。
ただし、以下の機能は使用できなくなります:
- 話者識別
- 話者登録
- 話者埋め込みファイルの生成

## 本番環境へのデプロイ

### フロントエンド

```bash
cd frontend
npm run build
```

ビルドされたファイルは `frontend/dist/` に出力されます。

### バックエンド

任意のASGIサーバーでホストします:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

または、Gunicorn + Uvicorn workers:

```bash
gunicorn backend.app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## ディレクトリ構成

```
minute_maker/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI アプリケーション
│   │   └── tests/                     # バックエンドテスト
│   ├── data/
│   │   ├── uploads/                   # アップロードされた音声ファイル
│   │   └── speakers/                  # 登録済み話者の埋め込みファイル
│   ├── tmp_model/                     # SpeechBrainモデルファイル（ダウンロード先）
│   └── requirements.txt               # Python依存関係
├── frontend/
│   ├── src/
│   │   ├── components/                # Reactコンポーネント
│   │   ├── App.tsx                    # メインアプリケーション
│   │   └── main.tsx                   # エントリーポイント
│   ├── public/                        # 静的アセット
│   └── package.json                   # Node依存関係とスクリプト
├── download_model.py                  # SpeechBrainモデルダウンロードスクリプト
├── .env                               # 環境変数（要作成）
├── AGENTS.md                          # 開発ログ
├── INSTALLATION.md                    # このファイル
└── README.md                          # プロジェクト概要
```

## サポート情報

- **SpeechBrain Documentation**: https://speechbrain.github.io/
- **Azure OpenAI Documentation**: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://react.dev/
