# exvs2ib-wiki-scraper

機動戦士ガンダム エクストリームバーサス2 インフィニットブースト（EXVS2IB）の非公式 Wiki から機体データを自動取得するスクレイパーツール。

**対象サイト**: [w.atwiki.jp/exvs2infiniteboost](https://w.atwiki.jp/exvs2infiniteboost)

## 取得データ

- 機体名・コスト・シリーズ・シリーズカテゴリ
- パイロット名
- 機体画像（ローカル保存）
- 詳細メタデータ（耐久値・BD回数 等）
- タグ

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### 全フェーズ一括実行

```bash
python scraper.py
```

### フェーズ個別実行

```bash
# Phase 1: 機体一覧を取得（output/units.json を生成）
python scraper.py --phase 1

# Phase 2: 各機体ページから詳細情報を取得
python scraper.py --phase 2

# Phase 3: 機体画像をダウンロード（output/images/ に保存）
python scraper.py --phase 3
```

### オプション一覧

| オプション | 説明 |
|-----------|------|
| `--phase {1,2,3}` | 実行フェーズを指定（省略時は全フェーズ実行） |
| `--force` | Phase 2 で処理済み機体も強制再取得する |
| `--no-images` | Phase 3 の画像ダウンロードをスキップ |
| `--output <path>` | 出力 JSON のパスを指定（config.yaml を上書き） |
| `--config <path>` | 設定ファイルのパスを指定（デフォルト: `config.yaml`） |
| `--verbose` | DEBUG レベルのログを出力 |
| `--dry-run` | 実際にリクエストせず URL リストを表示するのみ |

## 出力

| パス | 内容 |
|-----|------|
| `output/units.json` | 全機体データ（JSON） |
| `output/images/` | 機体画像（`{unitNo:03d}_{name}_{pageId}.png`） |
| `logs/scraper.log` | 実行ログ |

## 設定

`config.yaml` で以下の項目を変更できます。

```yaml
scraper:
  rate_limit: 1.0   # リクエスト間隔（秒）
  timeout: 10       # タイムアウト（秒）
  retry_count: 3    # リトライ回数
```

## 差分更新

Phase 2 は実行済み機体（`imageUrl` 取得済み）を自動スキップします。
強制的に全機体を再取得したい場合は `--force` を使用してください。

```bash
python scraper.py --phase 2 --force
```
