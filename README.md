# Minute Maker

FastAPI と Vite + React + TypeScript を組み合わせた議事録自動生成アプリです。音声や動画ファイルをアップロードすると、文字起こし・話者識別・要約を行い、Word / Excel 形式の議事録ファイルをダウンロードできます。

## アプリの概要

### 主要機能
- **音声・動画の文字起こし**: 
  - Azure OpenAI（GPT-4o / Whisper）による高精度な文字起こしと話者識別
  - OSS版Whisper（faster-whisper）による高速な文字起こし
- **話者識別**: SpeechBrainを使用した自動話者識別と話者登録機能
- **要約生成**: Azure OpenAI GPT-4oによる議事録の自動要約
- **議事録エクスポート**: Word / Excel 形式でのダウンロード
- **話者管理**:
  - 話者の登録・削除
  - 音声ファイルから話者埋め込みファイル(.npy)の生成・ダウンロード
- **ナビゲーション**:
  - 議事録作成、話者管理、プロンプト管理、設定画面の間をシームレスに切り替え
  - タブ切り替え時も作業内容を保持

### 処理フロー
1. **ファイルアップロード**: MP3/WAV/MP4/M4A 対応のドラッグ&ドロップ
2. **文字起こし**: 選択モデル（GPT-4o / Whisper）で音声をテキスト化
3. **話者識別**: 登録済み話者の自動識別
4. **確認・編集**: テーブル形式（開始・終了時間、話者、テキスト）で表示・編集
5. **要約・整形**: 要約プロンプトを選択して議事録を生成
6. **ダウンロード**: Word / Excel 形式でエクスポート

## クイックスタート

### 前提条件
- Python 3.12 以降
- Node.js 18 以降
- uv (Python パッケージマネージャー)

詳細なインストール手順は [INSTALLATION.md](INSTALLATION.md) を参照してください。

### 簡易セットアップ

#### 1. バックエンド
```bash
# 依存関係のインストール
uv sync

# 環境変数の設定（.envファイルを作成）
# AZURE_OPENAI_ENDPOINT=your_endpoint_here
# AZURE_OPENAI_API_KEY=your_api_key_here

# SpeechBrainモデルのダウンロード（重要！）
uv run python download_model.py

# サーバー起動
uv run uvicorn backend.app.main:app --reload
```

**OSS版Whisperを使用する場合:**
```bash
# .envファイルに以下を追加（Azure OpenAI設定は不要）
# WHISPER_PROVIDER=faster-whisper
# OSS_WHISPER_MODEL=base  # tiny/base/small/medium/large-v2/large-v3
# OSS_WHISPER_DEVICE=cpu  # または cuda（GPU使用時）

# 依存関係の再同期
uv sync

# サーバー起動
uv run uvicorn backend.app.main:app --reload
```

#### 2. フロントエンド
```bash
cd frontend
npm install
npm run dev -- --host
```

アプリケーションは `http://localhost:5173` で利用できます。

## 主要機能の詳細

### 1. 文字起こし
- **プロバイダー選択**: 
  - **Azure OpenAI**: GPT-4o（話者識別込み）またはWhisper（文字起こしのみ）
  - **OSS Whisper**: faster-whisper（文字起こしのみ、後からSpeechBrainで話者識別）
- **対応フォーマット**: MP3, WAV, MP4, M4A
- **タイムスタンプ**: セグメントごとの開始・終了時刻を記録

### 2. 話者管理
- **話者登録**: 
  - 文字起こし結果のセグメントから話者を登録
  - 音声ファイルから直接登録
- **話者識別**: 
  - 登録済み話者の自動検出
  - コサイン類似度による照合（閾値: 0.65）
- **埋め込みファイル生成**: 
  - 音声ファイルから話者特徴量（.npy）を生成・ダウンロード
  - システムに登録せずファイル生成のみも可能

### 3. 要約生成
- **プロンプト選択**:
  - 標準校正: バランスの取れた要約
  - 詳細: 背景情報と経緯を含む詳細分析
  - 簡潔: 要点のみの箇条書き
- **出力内容**:
  - 会議の要約
  - 決定事項
  - アクションアイテム

## API エンドポイント

### 議事録管理
- `GET /api/minutes` - 議事録一覧取得
- `POST /api/minutes` - 議事録作成
- `GET /api/minutes/{id}` - 議事録詳細取得
- `GET /api/minutes/{id}/download` - 議事録ダウンロード

### 音声処理
- `POST /api/process_audio` - 音声処理（文字起こし・話者識別）

### 話者管理
- `GET /api/speakers` - 登録済み話者一覧
- `POST /api/speakers` - 話者追加
- `DELETE /api/speakers/{name}` - 話者削除
- `POST /api/register_speaker` - セグメントから話者登録
- `POST /api/create_speaker_embedding` - 話者埋め込みファイル生成

### 要約生成
- `GET /api/prompts` - 要約プロンプト一覧
- `POST /api/generate_summary` - 要約生成

詳細は `http://localhost:8000/docs` のSwagger UIを参照してください。

## 使い方

### ナビゲーションバー
アプリケーション上部のナビゲーションバーで各機能画面に移動できます：
- **議事録作成**: メインの議事録作成ワークフロー
- **話者管理**: 話者の登録・削除、埋め込みファイル生成
- **プロンプト管理**: 要約用プロンプトの管理
- **設定**: LLMプロバイダーやWhisperモデルの選択

**重要**: タブを切り替えても、各画面の作業内容は保持されます。文字起こし処理中でも、設定変更や話者登録のために他のタブに移動し、後で「議事録作成」タブに戻って作業を続けることができます。

### 基本的な流れ
1. **議事録作成**タブを選択（デフォルトで表示）
2. 音声または動画ファイルをドラッグ&ドロップ
3. モデル（GPT-4o / Whisper）を選択（設定タブで事前に設定可能）
4. 「生成を開始する」をクリック
5. 処理完了後、文字起こし結果を確認・編集
6. 必要に応じて話者を登録（話者管理タブでも可能）
7. 要約プロンプトを選択して要約を生成
8. Word / Excelで議事録をダウンロード

### 話者登録の方法

#### 方法1: セグメントから登録
1. 文字起こし結果テーブルの「＋」アイコンをクリック
2. 話者名を入力
3. 「登録する」をクリック
4. 以降の処理で自動的に同じ声が識別されます

#### 方法2: 音声ファイルから登録
1. 「話者管理」ページに移動
2. 「新規登録」セクションで話者名と音声ファイルを選択
3. 「登録する」をクリック

#### 方法3: 埋め込みファイルの生成（登録なし）
1. 「話者管理」ページに移動
2. 「埋め込みファイル生成ツール」で音声ファイルを選択
3. 「生成してダウンロード」をクリック
4. `.npy`ファイルがダウンロードされます
5. このファイルは後で別のシステムで使用できます

## プロジェクト構成

```
minute_maker/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI アプリケーション
│   │   ├── azure_conversation_generation.py  # Azure OpenAI会話生成
│   │   └── tests/                     # バックエンドテスト
│   ├── data/
│   │   ├── uploads/                   # アップロードされた音声ファイル
│   │   └── speakers/                  # 登録済み話者の埋め込みファイル
│   ├── tmp_model/                     # SpeechBrainモデルファイル
│   └── requirements.txt               # Python依存関係
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── MinuteGenerator.tsx   # メイン議事録生成コンポーネント
│   │   │   └── SpeakerManager.tsx    # 話者管理コンポーネント
│   │   ├── App.tsx                    # トップレベルアプリケーション
│   │   └── main.tsx                   # エントリーポイント
│   ├── public/                        # 静的アセット
│   └── package.json                   # Node依存関係
├── download_model.py                  # SpeechBrainモデルダウンロードスクリプト
├── .env                               # 環境変数（要作成）
├── AGENTS.md                          # AI開発ログ
├── INSTALLATION.md                    # インストールガイド
└── README.md                          # このファイル
```

## 技術スタック

### バックエンド
- **FastAPI**: 高速なPython Webフレームワーク
- **Azure OpenAI**: GPT-4o / Whisper による文字起こし・要約（オプション）
- **faster-whisper**: OSS版Whisper（高速文字起こし、オプション）
- **SpeechBrain**: 話者認識（ECAPA-TDNN モデル）
- **PyTorch**: 機械学習フレームワーク
- **pydub**: 音声ファイル変換

### フロントエンド
- **React 18**: UIライブラリ
- **TypeScript**: 型安全性
- **Vite**: 高速ビルドツール
- **Lucide React**: アイコンライブラリ

## トラブルシューティング

### SpeechBrainモデルが読み込めない
1. `download_model.py` を実行してモデルをダウンロード
2. `backend/tmp_model/` に以下のファイルが存在することを確認:
   - `embedding_model.ckpt` (79.46 MB)
   - `classifier.ckpt` (5.28 MB)
   - `label_encoder.txt`
   - `hyperparams.yaml`

### 404エラーが発生する
- バックエンドとフロントエンドが両方起動していることを確認
- `VITE_API_BASE` 環境変数が正しく設定されていることを確認（デフォルト: `http://localhost:8000`）

### 話者識別が動作しない
1. SpeechBrainモデルが正しくダウンロードされていることを確認
2. 少なくとも1人の話者が登録されていることを確認
3. 音声セグメントが0.5秒以上であることを確認

詳細は [INSTALLATION.md](INSTALLATION.md) のトラブルシューティングセクションを参照してください。

## 本番ビルドとデプロイ

### フロントエンド
```bash
cd frontend
npm run build
```
ビルドされたファイルは `frontend/dist/` に出力されます。

### バックエンド
```bash
# Uvicorn（開発用）
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Gunicorn + Uvicorn workers（本番用）
gunicorn backend.app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## サポート

- **開発ログ**: [AGENTS.md](AGENTS.md)
- **インストールガイド**: [INSTALLATION.md](INSTALLATION.md)
- **API Documentation**: `http://localhost:8000/docs`

## 謝辞

以下のオープンソースプロジェクトを使用しています:
- [SpeechBrain](https://speechbrain.github.io/)
- [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
