#!/usr/bin/env python3
"""
EXVS2IB Wiki Scraper
機動戦士ガンダム エクストリームバーサス2 インフィニットブースト 非公式Wiki から
機体データを自動収集するスクレイパー
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from curl_cffi import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# 設定ロード
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ロギング設定
# ---------------------------------------------------------------------------

def setup_logging(cfg: dict) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "./logs/scraper.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("exvs2ib_scraper")
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# HTTP ユーティリティ
# ---------------------------------------------------------------------------

class RateLimitedSession:
    """レート制限付き HTTP セッション"""

    def __init__(self, cfg: dict, logger: logging.Logger):
        # impersonate="chrome" で TLS フィンガープリントをブラウザに偽装し Cloudflare を通過する
        self.session = requests.Session(impersonate="chrome")
        self.rate_limit: float = cfg["scraper"].get("rate_limit", 1.0)
        self.timeout: int = cfg["scraper"].get("timeout", 10)
        self.retry_count: int = cfg["scraper"].get("retry_count", 3)
        self.logger = logger
        self._last_request_time: float = 0.0

    def get(self, url: str) -> requests.Response:
        elapsed = time.time() - self._last_request_time
        wait = self.rate_limit - elapsed
        if wait > 0:
            time.sleep(wait)

        self.logger.debug("GET %s", url)
        resp = self.session.get(url, timeout=self.timeout)
        self._last_request_time = time.time()
        resp.raise_for_status()
        return resp


def fetch_soup(session: RateLimitedSession, url: str, logger: logging.Logger) -> Optional[BeautifulSoup]:
    try:
        resp = session.get(url)
        return BeautifulSoup(resp.content, "lxml")
    except requests.HTTPError as e:
        logger.warning("HTTP error %s: %s", e.response.status_code, url)
    except Exception as e:
        logger.warning("Fetch failed: %s — %s", url, e)
    return None


# ---------------------------------------------------------------------------
# Phase 1: 機体一覧テーブルパース
# ---------------------------------------------------------------------------

def _cell_text(cell) -> str:
    """
    縦書きレイアウト（<br/> で1文字ずつ区切り）セルのテキストを正規化して返す。
    - <br/> を '' に変換（文字を結合）
    - 連続する <br/><br/> を ' ' に変換（単語間スペース）
    - 縦書き長音符 '｜' (U+FF5C) → 'ー' (U+30FC) に置換
    """
    from bs4 import NavigableString, Tag
    parts: list[str] = []
    prev_was_br = False
    for child in cell.children:
        if isinstance(child, Tag) and child.name == "br":
            if prev_was_br:
                # 連続 <br/> → 区切りスペース
                parts.append(" ")
            prev_was_br = True
        else:
            if isinstance(child, NavigableString):
                t = str(child).strip()
                if t:
                    parts.append(t)
                    prev_was_br = False
            else:
                t = child.get_text(strip=True)
                if t:
                    parts.append(t)
                    prev_was_br = False
    text = "".join(parts).replace("｜", "ー").strip()
    # 複数スペースを1つに圧縮
    import re as _re
    return _re.sub(r"\s+", " ", text)

def _normalize_cost(text: str) -> Optional[str]:
    """ヘッダーテキストからコスト数値文字列を抽出する（例: '3000コスト' → '3000'）"""
    m = re.search(r"(3000|2500|2000|1500)", text)
    return m.group(1) if m else None


def _parse_unit_list_table(soup: BeautifulSoup, base_url: str, target_costs: list[str], logger: logging.Logger) -> list[dict]:
    """
    機体一覧テーブルをパースし、対象コスト列の機体リストを返す。
    rowspan を正確に追跡してシリーズカテゴリ・シリーズ名を付与する。

    Returns:
        list of {name, pageId, wikiUrl, cost, series, seriesCategory, isNew, costChanged}
        ※ unitNo はまだ付与しない（採番は呼び出し側で行う）
    """
    table = soup.select_one("div.center_plugin table")
    if table is None:
        logger.error("機体一覧テーブルが見つかりません (div.center_plugin table)")
        return []

    rows = table.select("tr")
    if not rows:
        logger.error("テーブルに行がありません")
        return []

    # ---- ヘッダー行からコスト列インデックスをマッピング ----
    header_row = rows[0]
    header_cells = header_row.find_all(["th", "td"])
    cost_col_map: dict[int, str] = {}  # col_index → cost ("3000" etc.)
    for i, cell in enumerate(header_cells):
        text = cell.get_text(strip=True)
        cost = _normalize_cost(text)
        if cost and cost in target_costs:
            cost_col_map[i] = cost

    if not cost_col_map:
        logger.error("対象コスト列が見つかりません。ヘッダー: %s", [c.get_text(strip=True) for c in header_cells])
        return []
    logger.info("コスト列インデックス: %s", cost_col_map)

    # ---- rowspan トラッカー ----
    # rowspan_tracker[col_index] = {"remaining": N, "value": str}
    rowspan_tracker: dict[int, dict] = {}

    # コスト列ごとに採集した機体リスト（順序保持のため cost → list）
    cost_units: dict[str, list[dict]] = {c: [] for c in target_costs}

    # シリーズカテゴリ列（インデックス 0）とシリーズ名列（インデックス 1）を追跡
    series_cat_tracker = {"remaining": 0, "value": ""}
    series_tracker = {"remaining": 0, "value": ""}

    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if not cells:
            continue

        # rowspan_tracker を1行消費
        def consume_tracker(tracker: dict) -> str:
            if tracker["remaining"] > 0:
                tracker["remaining"] -= 1
                return tracker["value"]
            return ""

        # シリーズカテゴリとシリーズ名をセルから読み取るか、rowspan から継続するか判断
        cell_iter = iter(cells)

        # カラム 0: シリーズカテゴリ（rowspan あり）
        if series_cat_tracker["remaining"] > 0:
            series_cat = series_cat_tracker["value"]
            series_cat_tracker["remaining"] -= 1
        else:
            try:
                c = next(cell_iter)
                series_cat = _cell_text(c)
                rs = int(c.get("rowspan", 1))
                series_cat_tracker = {"remaining": rs - 1, "value": series_cat}
            except StopIteration:
                continue

        # カラム 1: シリーズ名（rowspan あり）
        if series_tracker["remaining"] > 0:
            series_name = series_tracker["value"]
            series_tracker["remaining"] -= 1
        else:
            try:
                c = next(cell_iter)
                series_name = _cell_text(c)
                rs = int(c.get("rowspan", 1))
                series_tracker = {"remaining": rs - 1, "value": series_name}
            except StopIteration:
                continue

        # 残りのセルをコスト列に対応させる
        remaining_cells = list(cell_iter)

        # rowspan_tracker を参照しながら実際のカラムインデックスを解決する
        # ヘッダーのカラム数（シリーズカテゴリ2列 + コスト列群）
        # インデックスは 0-based でヘッダーに合わせる
        # col 0,1 はシリーズ列として消費済み。残りを col 2 から割り当てる
        # rowspan_tracker で「スキップすべき列」を管理
        col_cursor = 2  # シリーズ2列の次から
        cell_idx = 0
        col_to_cell: dict[int, any] = {}

        max_col = max(cost_col_map.keys()) + 1
        while col_cursor < max_col:
            if col_cursor in rowspan_tracker and rowspan_tracker[col_cursor]["remaining"] > 0:
                # このカラムは前行の rowspan で埋まっている → セルを消費しない
                rowspan_tracker[col_cursor]["remaining"] -= 1
                col_cursor += 1
                continue
            if cell_idx < len(remaining_cells):
                cell = remaining_cells[cell_idx]
                col_to_cell[col_cursor] = cell
                rs = int(cell.get("rowspan", 1))
                if rs > 1:
                    rowspan_tracker[col_cursor] = {"remaining": rs - 1, "value": None}
                cell_idx += 1
            col_cursor += 1

        # 対象コスト列のセルを処理
        for col_idx, cost in cost_col_map.items():
            cell = col_to_cell.get(col_idx)
            if cell is None:
                continue

            links = cell.select('a[href*="/pages/"]')
            if not links:
                continue

            cell_text = cell.get_text()
            is_new = bool(cell.select('span[style*="color: RED"]')) or "NEW!" in cell_text
            cost_changed = "↑" in cell_text or "↓" in cell_text

            for a in links:
                href = a.get("href", "")
                # href="//w.atwiki.jp/exvs2infiniteboost/pages/XXX.html"
                m = re.search(r"/pages/(\d+)\.html", href)
                if not m:
                    continue
                page_id = m.group(1)
                name = a.get_text(strip=True)
                # NEW!/↑↓ マークを機体名から除去
                name = re.sub(r"NEW!|↑|↓", "", name).strip()
                if not name:
                    continue

                wiki_url = f"https://w.atwiki.jp/exvs2infiniteboost/pages/{page_id}.html"
                cost_units[cost].append({
                    "name": name,
                    "pageId": page_id,
                    "wikiUrl": wiki_url,
                    "cost": cost,
                    "series": series_name,
                    "seriesCategory": series_cat,
                    "isNew": is_new,
                    "costChanged": cost_changed,
                })

    # ---- unitNo 採番: コスト降順 × 出現順 ----
    ordered_costs = ["3000", "2500", "2000", "1500"]
    units: list[dict] = []
    unit_no = 1
    for cost in ordered_costs:
        for unit in cost_units.get(cost, []):
            unit["unitNo"] = unit_no
            units.append(unit)
            unit_no += 1

    return units


def phase1(cfg: dict, session: RateLimitedSession, logger: logging.Logger, dry_run: bool = False) -> list[dict]:
    """Phase 1: 機体一覧ページから全機体のメタ情報を抽出し JSON 出力する"""
    base_url = cfg["scraper"]["base_url"]
    index_url = base_url + cfg["scraper"]["index_page"]
    target_costs: list[str] = [str(c) for c in cfg.get("target_columns", ["3000", "2500", "2000", "1500"])]

    logger.info("=== Phase 1: 機体一覧取得 ===")
    logger.info("URL: %s", index_url)

    soup = fetch_soup(session, index_url, logger)
    if soup is None:
        logger.error("機体一覧ページの取得に失敗しました")
        sys.exit(1)

    units = _parse_unit_list_table(soup, base_url, target_costs, logger)
    logger.info("抽出機体数: %d", len(units))

    if dry_run:
        logger.info("[dry-run] Phase 1 完了。出力ファイルには書き込みません")
        for u in units:
            logger.info("  No.%03d %s (cost=%s, series=%s)", u["unitNo"], u["name"], u["cost"], u["series"])
        return units

    out_path = Path(cfg["output"]["json_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1 結果を保存（後続フェーズのキャッシュとして使用）
    output = {
        "units": units,
        "metadata": {
            "scrapedAt": datetime.now(timezone.utc).isoformat(),
            "sourceUrl": index_url,
            "totalUnits": len(units),
            "scraperVersion": __version__,
            "phase": 1,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Phase 1 出力: %s", out_path)

    return units


# ---------------------------------------------------------------------------
# JSON キャッシュ ユーティリティ
# ---------------------------------------------------------------------------

def load_cached_units(cfg: dict, logger: logging.Logger) -> list[dict]:
    """既存の units.json を読み込む（中間キャッシュ）"""
    out_path = Path(cfg["output"]["json_path"])
    if not out_path.exists():
        logger.error("キャッシュが見つかりません: %s  先に Phase 1 を実行してください", out_path)
        sys.exit(1)
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    units = data.get("units", [])
    logger.info("キャッシュ読み込み: %d 機体", len(units))
    return units


def save_units(cfg: dict, units: list[dict], phase: int, logger: logging.Logger) -> None:
    """units を JSON に保存する"""
    out_path = Path(cfg["output"]["json_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # scrapedAt は既存があれば引き継ぐ
    scraped_at = datetime.now(timezone.utc).isoformat()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        scraped_at = existing.get("metadata", {}).get("scrapedAt", scraped_at)

    output = {
        "units": units,
        "metadata": {
            "scrapedAt": scraped_at,
            "sourceUrl": cfg["scraper"]["base_url"] + cfg["scraper"]["index_page"],
            "totalUnits": len(units),
            "scraperVersion": __version__,
            "phase": phase,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Phase %d 出力: %s", phase, out_path)


# ---------------------------------------------------------------------------
# Phase 2: 個別ページから基本情報取得
# ---------------------------------------------------------------------------

# 一覧ページのメタデータテーブルキー → JSONフィールド名マッピング
_META_KEY_MAP = {
    "作品枠": "_seriesFromPage",
    "パイロット": "pilot",
    "コスト": "_costFromPage",
    "耐久値": "durability",
    "形態移行": "formChange",
    "移動タイプ": "moveType",
    "BD回数": "bdCount",
    "赤ロック距離": "redLockRange",
    "変形コマンド": "_hasTransformRaw",
    "盾コマンド": "_hasShieldRaw",
    "扱いやすさ": "difficulty",
    "デフォルトBGM": "defaultBGM",
}


def _parse_unit_page(soup: BeautifulSoup, unit: dict, logger: logging.Logger) -> dict:
    """個別ページから基本情報を抽出し unit dict を更新して返す（Phase 2）"""
    page_id = unit["pageId"]

    # ページタイトル（機体名確認用）
    title_el = soup.select_one("#wikibody h2 a") or soup.select_one("#wikibody h2")
    page_title = title_el.get_text(strip=True) if title_el else ""

    # メタデータテーブル（div.float-right 内）
    meta_table = soup.select_one("div.float-right table")
    page_series = ""
    page_cost = ""
    pilot = unit.get("pilot", "")

    if meta_table:
        for row in meta_table.select("tr"):
            tds = row.select("td")
            if len(tds) >= 2:
                key = tds[0].get_text(strip=True)
                val = tds[1].get_text(strip=True)
                if key == "作品枠":
                    page_series = val
                elif key == "パイロット":
                    pilot = val
                elif key == "コスト":
                    page_cost = val

    # series 差異チェック（一覧ページ優先）
    list_series = unit.get("series", "")
    if page_series and page_series != list_series:
        logger.warning(
            "[series diff] No.%03d %s (pageId=%s): 一覧='%s' 個別ページ='%s' → 一覧を採用",
            unit["unitNo"], unit["name"], page_id, list_series, page_series,
        )

    # cost 差異チェック（一覧ページ優先）
    list_cost = unit.get("cost", "")
    if page_cost and page_cost != list_cost:
        logger.warning(
            "[cost diff] No.%03d %s (pageId=%s): 一覧='%s' 個別ページ='%s' → 一覧を採用",
            unit["unitNo"], unit["name"], page_id, list_cost, page_cost,
        )

    unit["pilot"] = pilot
    return unit


def phase2(cfg: dict, session: RateLimitedSession, logger: logging.Logger, dry_run: bool = False) -> list[dict]:
    """
    Phase 2: 各機体ページから全情報を1リクエストで一括取得する。
    - パイロット（top-level）
    - 画像URL（imgタグ → OGP meta フォールバック）
    - 詳細メタデータ（耐久値・BD回数 等）
    - タグ
    - series / cost 差異チェック（一覧ページ優先、差異はログに記録）
    """
    logger.info("=== Phase 2: 個別ページ全情報取得 ===")
    units = load_cached_units(cfg, logger)
    no_image_units: list[str] = []

    for unit in tqdm(units, desc="Phase2", unit="機体"):
        soup = fetch_soup(session, unit["wikiUrl"], logger)
        if soup is None:
            logger.warning("スキップ: %s", unit["wikiUrl"])
            unit.setdefault("pilot", "")
            unit.setdefault("imageUrl", None)
            unit.setdefault("imageLocalPath", None)
            unit.setdefault("tags", [])
            unit.setdefault("metadata", {})
            continue

        # 基本情報（pilot・series/cost 差異チェック）
        _parse_unit_page(soup, unit, logger)

        # 画像URL
        image_url = _extract_image_url(soup)
        unit["imageUrl"] = image_url
        unit["imageLocalPath"] = None
        if image_url is None:
            no_image_units.append(f"No.{unit['unitNo']:03d} {unit['name']}")
            logger.warning("画像なし: No.%03d %s (pageId=%s)", unit["unitNo"], unit["name"], unit["pageId"])

        # 詳細メタデータ・タグ
        _parse_metadata_and_tags(soup, unit, logger)

    if no_image_units:
        logger.info("画像が見つからなかった機体 %d 件: %s", len(no_image_units), no_image_units)

    if not dry_run:
        save_units(cfg, units, phase=2, logger=logger)
    return units


# ---------------------------------------------------------------------------
# 画像URL抽出ヘルパー（Phase 2 内で使用）
# ---------------------------------------------------------------------------

def _extract_image_url(soup: BeautifulSoup) -> Optional[str]:
    """div.float-right の img タグ → OGP meta の順で画像URLを取得する"""
    img = soup.select_one("div.float-right img.atwiki_plugin_image")
    if img and img.get("src"):
        src = img["src"]
        return ("https:" + src) if src.startswith("//") else src

    og = soup.select_one('meta[property="og:image"]')
    if og and og.get("content"):
        return og["content"]

    return None


# ---------------------------------------------------------------------------
# 詳細メタデータ・タグ抽出ヘルパー（Phase 2 内で使用）
# ---------------------------------------------------------------------------

def _parse_bool_field(val: str) -> bool:
    """'なし' / 空文字 → False、それ以外 → True"""
    return bool(val) and val not in ("なし", "-", "—")


def _parse_metadata_and_tags(soup: BeautifulSoup, unit: dict, logger: logging.Logger) -> dict:
    """詳細メタデータとタグを抽出して unit を更新する"""
    page_id = unit["pageId"]
    meta: dict = {}

    meta_table = soup.select_one("div.float-right table")
    if meta_table:
        for row in meta_table.select("tr"):
            tds = row.select("td")
            if len(tds) >= 2:
                key = tds[0].get_text(strip=True)
                val = tds[1].get_text(strip=True)
                json_key = _META_KEY_MAP.get(key)
                if json_key and not json_key.startswith("_"):
                    meta[json_key] = val
                elif key == "変形コマンド":
                    meta["hasTransform"] = _parse_bool_field(val)
                elif key == "盾コマンド":
                    meta["hasShield"] = _parse_bool_field(val)

    # pilot は top-level に存在するので metadata には入れない
    meta.pop("pilot", None)

    # タグ
    tags = [
        a.get_text(strip=True)
        for a in soup.select("div.atwiki-tags-wrap a.atwiki-tag-frame")
    ]

    unit["tags"] = tags
    unit["metadata"] = meta
    return unit


# ---------------------------------------------------------------------------
# Phase 3: 画像ダウンロード
# ---------------------------------------------------------------------------

_INVALID_CHARS = re.compile(r'[/\\:*?"<>|]')


def _safe_filename(name: str) -> str:
    return _INVALID_CHARS.sub("_", name)


def phase3(cfg: dict, session: RateLimitedSession, logger: logging.Logger, dry_run: bool = False) -> list[dict]:
    """Phase 3: 画像をローカルに保存する（Phase 2 で取得した imageUrl を使用）"""
    logger.info("=== Phase 3: 画像ダウンロード ===")
    units = load_cached_units(cfg, logger)
    image_dir = Path(cfg["output"]["image_dir"])
    image_dir.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = failed = 0

    for unit in tqdm(units, desc="Phase3", unit="機体"):
        image_url = unit.get("imageUrl")
        if not image_url:
            unit["imageLocalPath"] = None
            continue

        safe_name = _safe_filename(unit["name"])
        filename = f"{unit['unitNo']:03d}_{safe_name}_{unit['pageId']}.png"
        local_path = image_dir / filename
        relative_path = f"./output/images/{filename}"

        if local_path.exists():
            logger.debug("スキップ（既存）: %s", filename)
            unit["imageLocalPath"] = relative_path
            skipped += 1
            continue

        if dry_run:
            unit["imageLocalPath"] = relative_path
            continue

        try:
            resp = session.get(image_url)
            local_path.write_bytes(resp.content)
            unit["imageLocalPath"] = relative_path
            downloaded += 1
            logger.debug("保存: %s", filename)
        except Exception as e:
            logger.warning("画像DL失敗: %s — %s", image_url, e)
            unit["imageLocalPath"] = None
            failed += 1

    logger.info("画像DL完了: %d 件 / スキップ: %d 件 / 失敗: %d 件", downloaded, skipped, failed)

    if not dry_run:
        save_units(cfg, units, phase=3, logger=logger)
    return units


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXVS2IB Wiki スクレイパー")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルパス")
    parser.add_argument("--no-images", action="store_true", help="画像ダウンロードをスキップ")
    parser.add_argument("--output", help="出力JSONパス（config.yaml を上書き）")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ出力")
    parser.add_argument("--dry-run", action="store_true", help="実行せずURLリスト表示のみ")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="実行フェーズ指定: 1=一覧取得 2=個別ページ全情報 3=画像DL（省略時は全フェーズ実行）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.verbose:
        cfg.setdefault("logging", {})["level"] = "DEBUG"
    if args.output:
        cfg.setdefault("output", {})["json_path"] = args.output
    if args.no_images:
        cfg.setdefault("output", {})["download_images"] = False

    logger = setup_logging(cfg)
    logger.info("exvs2ib-wiki-scraper v%s", __version__)

    session = RateLimitedSession(cfg, logger)

    phases = [args.phase] if args.phase else [1, 2, 3]

    for phase_no in phases:
        if phase_no == 1:
            phase1(cfg, session, logger, dry_run=args.dry_run)
        elif phase_no == 2:
            phase2(cfg, session, logger, dry_run=args.dry_run)
        elif phase_no == 3:
            if cfg["output"].get("download_images", True):
                phase3(cfg, session, logger, dry_run=args.dry_run)
            else:
                logger.info("Phase 3 スキップ（--no-images）")

    logger.info("完了")


if __name__ == "__main__":
    main()
