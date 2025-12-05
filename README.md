# Minute Maker

FastAPI と Vite + React + TypeScript を組み合わせた議事録自動生成アプリです。音声や動画ファイルをアップロードすると、文字起こし・話者識別・要約を行い、Word / Excel 形式の議事録ファイルをダウンロードできます。

## アプリの概要
- **議事録生成フロー**: ファイルアップロード → 文字起こし → 話者識別 → 要約 → ダウンロード。
- **UI**: MinuteGenerator コンポーネントでドラッグ＆ドロップのアップロード、処理ステップの進行表示、要約結果・全文テーブル表示を提供します。
- **機能**:
    - 文字起こし結果をテーブル形式（開始・終了時間、話者、テキスト）で表示。
    - 任意の行の音声データから話者を登録し、以後の識別でその名前を使用可能。
- **API**: 
    - `/api/process_audio`: 音声処理（文字起こし・話者識別）
    - `/api/register_speaker`: 話者登録（音声セグメントから埋め込み作成）
    - `/api/minutes`: 議事録データの取得・追加

## クイックスタート
### 前提
- Python 3.11 以降
- Node.js 18 以降

### バックエンドの起動 (FastAPI)
1. 依存関係のインストール（仮想環境推奨）:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. 開発サーバー起動:
   ```bash
   uvicorn backend.app.main:app --reload
   ```
3. 確認エンドポイント:
   - ヘルスチェック: `GET /`
   - 議事録一覧・登録: `GET/POST /api/minutes`
   - 議事録詳細: `GET /api/minutes/{id}`
   - 音声処理: `POST /api/process_audio`
   - 話者登録: `POST /api/register_speaker`
   - 議事録ダウンロード（モック）: `GET /api/minutes/{id}/download?format=docx|xlsx`

### フロントエンドの起動 (Vite + React + TypeScript)
1. 依存関係のインストール:
   ```bash
   cd frontend
   npm install
   ```
2. 開発サーバー起動:
   ```bash
   npm run dev -- --host
   ```
3. API の接続先は `VITE_API_BASE` 環境変数で指定できます（デフォルト: `http://localhost:8000`）。

### 使い方
1. フロントエンドにアクセスし、「作成」タブで音声または動画ファイルをドラッグ＆ドロップ（MP3/WAV/MP4/M4A 対応）。
2. 「生成を開始する」をクリックすると処理ステップが進行し、完了後に要約結果と全文テーブルが表示されます。
3. **話者登録機能**:
   - テーブルの各行にある「＋」アイコン（話者登録ボタン）をクリックします。
   - そのセグメントの話者名を入力して「登録する」を押すと、その声の特徴がシステムに保存されます。
   - 以降のファイル処理時に、同じ声が検出されると自動的に登録された名前が表示されます。
4. Word (`.docx`) か Excel (`.xlsx`) のいずれかで議事録をダウンロードできます。

### 本番ビルドとデプロイ
- フロントエンドビルド: `cd frontend && npm run build`
- バックエンド: 任意の ASGI サーバーで `backend.app.main:app` をホストします（例: `uvicorn backend.app.main:app`）。

## プロジェクト構成
```
backend/
  app/main.py         # FastAPI アプリケーションとメモリ上のデータ
  requirements.txt    # Python 依存関係
frontend/
  public/             # 静的アセット
  src/                # React アプリケーション
  package.json        # Node 依存関係とスクリプト
```
