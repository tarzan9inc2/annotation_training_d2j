# セットアップ手順

## 1. Pythonのインストール

### 推奨バージョン
**Python 3.10** を強く推奨します（最も安定）

- ✅ **Python 3.10.11** - **最もおすすめ** (安定性とパフォーマンスのバランスが良い)
- ✅ **Python 3.9** - 安定版（少し古いが非常に安定）
- ⚠️ **Python 3.11** - 動作するが、一部パッケージで問題が出る可能性
- ❌ **Python 3.12以降** - PyTorchやその他パッケージの対応が不完全

### Windows
1. [Python 3.10.11 ダウンロード](https://www.python.org/downloads/release/python-31011/)
   - Windows用インストーラー: "Windows installer (64-bit)" を選択
2. インストーラーを実行
3. **重要**: "Add Python to PATH" に必ずチェックを入れる
4. "Install Now" をクリック
5. インストール完了後、コンピュータを再起動（推奨）

### インストール確認
コマンドプロンプトで以下を実行：
```cmd
python --version
```
または
```cmd
py --version
```

## 2. セットアップ（初回のみ）

### setup.bat を実行
初回のみ、以下を実行してください：

1. `setup.bat` をダブルクリック
2. または、コマンドプロンプトで：
```cmd
setup.bat
```

セットアップでは以下の処理が行われます：
- 既存の仮想環境の削除（もしあれば）
- 新しい仮想環境（venv）の作成
- 必要なパッケージのインストール（requirements.txtから）

**注意**: インストールに5〜10分程度かかる場合があります（特にPyTorchは大きいファイルです）。

## 3. アプリケーションの起動

### 起動方法
セットアップ完了後、以下で起動できます：

1. `run.bat` をダブルクリック
2. または、コマンドプロンプトで：
```cmd
run.bat
```

### 2回目以降
毎回 `run.bat` をダブルクリックするだけで起動します。

## 4. トラブルシューティング

### Python が見つからない
- Pythonが正しくインストールされているか確認
- 環境変数のPATHにPythonが含まれているか確認
- コンピュータを再起動してPATHを更新

### インストールエラー
- インターネット接続を確認
- 管理者権限で実行してみる
- `venv` フォルダを削除して再実行

### パッケージのインストールに失敗
特定のパッケージ（torch、torchvisionなど）は大きいため、インストールに時間がかかります。
タイムアウトする場合は、以下を手動で実行：

```cmd
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 必要な依存パッケージ

- PyQt5 (GUI)
- PyTorch (機械学習)
- torchvision (画像処理)
- Pillow (画像読み込み)
- NumPy (数値計算)
- その他（requirements.txt参照）

## システム要件

- Python 3.7以上
- Windows 10/11
- RAM: 8GB以上推奨
- GPU: CUDA対応GPU推奨（CPUでも動作可能）
