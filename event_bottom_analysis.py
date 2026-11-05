#!/usr/bin/env python3
"""重大イベント日を起点に「何営業日後に底を付け、何日で起点水準に戻ったか」を検証する。

使い方の例:
    python event_bottom_analysis.py --ticker ^GSPC ^NDX NVDA SOXX
    python event_bottom_analysis.py --ticker ^GSPC --events my_events.csv --window 60 --min-drop -5

定義（ルックアヘッド・バイアス回避のため、起点は起点日以前のデータだけで決める）:
    イベント日 E（暦日）。
    起点 = E より前の最後の取引日の終値（「イベント前日終値」）。
    観測期間 = E 以降の最初の取引日から window 営業日。
    底 = 観測期間内の最安終値。
    回復 = 底の後、終値が起点終値を初めて上回った日（観測期間内のみ）。
    直前1か月/12か月リターン = 起点日までの 21 / 252 営業日リターン。
    景気後退期 = FRED USREC（NBER 基準）の起点月の値。NBER の判定は事後公表なので、
                 これは「当時わかっていた情報」ではなく、事後の分類として使う。
"""
from __future__ import annotations

import argparse
import io
import sys
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

SMALL_SAMPLE = 10  # これ未満の集計には警告を付ける
GAP_DAYS = 10      # 連続する終値の間がこの暦日数を超えたら「データ欠損の疑い」として報告

# stooq のティッカー表記（不明なものは米国株とみなして .us を付ける）
STOOQ_MAP = {"^GSPC": "^spx", "^NDX": "^ndx", "^DJI": "^dji", "^IXIC": "^ndq"}
# FRED に日次系列があるもの（FRED の SP500 等は直近約10年分しか公開されていない点に注意）
FRED_MAP = {"^GSPC": "SP500", "^NDX": "NASDAQ100", "^IXIC": "NASDAQCOM", "^DJI": "DJIA"}


# --------------------------------------------------------------------------- イベント
def election_days(start_year: int = 1980, end_year: int | None = None) -> list[tuple[date, str]]:
    """米国の連邦選挙日（11月の第1月曜日の翌日の火曜日）を計算で求める。偶数年のみ。"""
    end_year = end_year or date.today().year
    out = []
    for y in range(start_year, end_year + 1):
        if y % 2:
            continue
        d = date(y, 11, 1)
        while d.weekday() != 0:  # 第1月曜日
            d += timedelta(days=1)
        tue = d + timedelta(days=1)
        kind = "大統領選" if y % 4 == 0 else "中間選挙"
        out.append((tue, f"{y} {kind}"))
    return out


# 地政学ショック（イベント発生日。市場休場中に起きた場合も、E 以降の最初の取引日から観測する）
GEOPOLITICAL = [
    ("1990-08-02", "イラクのクウェート侵攻"),
    ("2001-09-11", "米同時多発テロ(9/11)"),
    ("2003-03-20", "イラク侵攻開始"),
    ("2022-02-24", "ロシアのウクライナ侵攻"),
]


def default_events() -> pd.DataFrame:
    rows = []
    for d, label in election_days():
        # 「選挙の翌営業日」をイベント反応日とする: E = 選挙日の翌暦日。
        # 起点 = E より前の最後の終値 = 選挙当日の終値。
        rows.append({"date": pd.Timestamp(d + timedelta(days=1)), "label": label + "（翌営業日）",
                     "category": "選挙"})
    for d, label in GEOPOLITICAL:
        rows.append({"date": pd.Timestamp(d), "label": label, "category": "地政学"})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_events(path: str | None) -> pd.DataFrame:
    if not path:
        return default_events()
    ev = pd.read_csv(path)
    missing = {"date", "label"} - set(ev.columns)
    if missing:
        sys.exit(f"イベントCSVに必要な列がありません: {missing}")
    ev["date"] = pd.to_datetime(ev["date"])
    if "category" not in ev.columns:
        ev["category"] = "ユーザー指定"
    return ev[["date", "label", "category"]].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- データ取得
@dataclass
class PriceSeries:
    ticker: str
    close: pd.Series        # DatetimeIndex -> 終値
    source: str
    note: str = ""


def _clean(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s[s > 0]


def fetch_yfinance(ticker: str, start: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance: 空のデータ")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return _clean(close)


def fetch_stooq(ticker: str) -> pd.Series:
    import requests
    sym = STOOQ_MAP.get(ticker.upper(), ticker.lower() if ticker.startswith("^") else f"{ticker.lower()}.us")
    r = requests.get("https://stooq.com/q/d/l/", params={"s": sym, "i": "d"}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Close" not in df.columns or df.empty:
        raise RuntimeError(f"stooq: データなし ({sym}): {r.text[:80]!r}")
    return _clean(df.set_index("Date")["Close"])


def fetch_fred_series(series_id: str) -> pd.Series:
    import requests
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    return pd.to_numeric(df.set_index(date_col)[series_id], errors="coerce").dropna().pipe(
        lambda s: s.set_axis(pd.to_datetime(s.index)))


def load_prices(ticker: str, start: str, price_dir: str | None) -> tuple[PriceSeries | None, list[str]]:
    """yfinance -> stooq -> FRED の順に試す。補完や推測はしない。"""
    log = []
    if price_dir:
        p = Path(price_dir) / f"{ticker.replace('^', '')}.csv"
        if p.exists():
            df = pd.read_csv(p)
            return PriceSeries(ticker, _clean(df.set_index(df.columns[0])["Close"]), f"ローカルCSV ({p})"), log
        log.append(f"ローカルCSVなし: {p}")
    try:
        return PriceSeries(ticker, fetch_yfinance(ticker, start), "yfinance（調整後終値）"), log
    except Exception as e:  # noqa: BLE001
        log.append(f"yfinance 失敗: {type(e).__name__}: {str(e)[:120]}")
    try:
        return PriceSeries(ticker, fetch_stooq(ticker), "stooq（株式分割調整済み・配当は未調整の可能性）"), log
    except Exception as e:  # noqa: BLE001
        log.append(f"stooq 失敗: {type(e).__name__}: {str(e)[:120]}")
    if ticker.upper() in FRED_MAP:
        try:
            s = fetch_fred_series(FRED_MAP[ticker.upper()])
            return PriceSeries(ticker, _clean(s), f"FRED {FRED_MAP[ticker.upper()]}（価格指数・配当なし）",
                               "FREDの公開期間外のイベントは欠損"), log
        except Exception as e:  # noqa: BLE001
            log.append(f"FRED 失敗: {type(e).__name__}: {str(e)[:120]}")
    else:
        log.append("FRED: 対応する系列なし")
    return None, log


def load_recession(price_dir: str | None) -> tuple[pd.Series | None, str]:
    if price_dir:
        p = Path(price_dir) / "USREC.csv"
        if p.exists():
            df = pd.read_csv(p)
            s = df.set_index(df.columns[0]).iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            return s, f"ローカルCSV ({p})"
    try:
        return fetch_fred_series("USREC"), "FRED USREC"
    except Exception as e:  # noqa: BLE001
        return None, f"取得失敗（{type(e).__name__}）。景気後退期の判定は欠損として扱う"


# --------------------------------------------------------------------------- 指標計算
def analyze_event(px: pd.Series, ev_date: pd.Timestamp, window: int, rec: pd.Series | None) -> dict:
    out: dict = {"status": "ok"}
    before = px[px.index < ev_date]          # 起点はイベント日より前のデータだけで決める
    after = px[px.index >= ev_date]
    if before.empty or (before.index[-1] < ev_date - pd.Timedelta(days=GAP_DAYS)):
        out["status"] = "欠損: 起点日のデータなし"
        return out
    o_date, o_close = before.index[-1], float(before.iloc[-1])
    out.update(origin_date=o_date.date(), origin_close=o_close)

    # 直前リターン（起点日までのデータのみ使用）
    for n, name in [(21, "ret_1m_before"), (252, "ret_12m_before")]:
        out[name] = (o_close / float(before.iloc[-1 - n]) - 1) * 100 if len(before) > n else np.nan

    # 景気後退期（起点月）
    if rec is not None:
        m = rec[rec.index <= o_date]
        out["recession"] = (int(m.iloc[-1]) == 1) if not m.empty and m.index[-1] >= o_date - pd.Timedelta(days=62) else None
    else:
        out["recession"] = None

    win = after.iloc[:window]
    out["window_days_available"] = len(win)
    if win.empty:
        out["status"] = "欠損: イベント後のデータなし"
        return out
    out["window_complete"] = len(win) == window
    seg = pd.concat([before.iloc[-1:], win])
    gaps = seg.index.to_series().diff().dt.days
    out["max_gap_calendar_days"] = int(gaps.max()) if len(gaps) > 1 else 0
    out["data_gap_flag"] = bool(out["max_gap_calendar_days"] > GAP_DAYS)

    b_pos = int(np.argmin(win.values))
    b_date, b_close = win.index[b_pos], float(win.iloc[b_pos])
    out.update(
        bottom_date=b_date.date(), bottom_close=b_close,
        days_to_bottom_trading=b_pos + 1,
        days_to_bottom_calendar=(b_date - o_date).days,
        max_drop_pct=(b_close / o_close - 1) * 100,
        bottom_at_window_end=(b_pos == len(win) - 1),  # 観測期間の末尾が最安値 = 底は未確定（打ち切り）
    )
    if b_close >= o_close:
        out["recovery"] = "下落なし"
        out["recovery_days_trading"] = 0
        out["recovery_days_calendar"] = 0
    else:
        later = win.iloc[b_pos + 1:]
        hit = later[later > o_close]
        if hit.empty:
            out["recovery"] = "未回復" if out["window_complete"] else "未回復（観測期間がデータ末尾で途切れ）"
            out["recovery_days_trading"] = np.nan
            out["recovery_days_calendar"] = np.nan
        else:
            r_date = hit.index[0]
            out["recovery"] = "回復"
            out["recovery_date"] = r_date.date()
            out["recovery_days_trading"] = int(win.index.get_loc(r_date)) + 1
            out["recovery_days_calendar"] = (r_date - o_date).days
    if not out["window_complete"]:
        out["status"] = f"不完全: 観測期間 {len(win)}/{window} 営業日（集計から除外）"
    return out


# --------------------------------------------------------------------------- 集計
def describe(s: pd.Series, worst_is_min: bool = False) -> dict:
    """worst_is_min=True は下落率用（最も深い下落 = 最小値を「最大」として出す）。"""
    s = s.dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "mean": s.mean(), "median": s.median(),
            "p25": s.quantile(0.25), "p75": s.quantile(0.75),
            "max": s.min() if worst_is_min else s.max()}


def fmt_stats(d: dict, unit: str = "日") -> str:
    if d.get("n", 0) == 0:
        return "標本なし"
    w = " ⚠️標本10件未満" if d["n"] < SMALL_SAMPLE else ""
    return (f"n={d['n']}{w} / 平均 {d['mean']:.1f}{unit} / 中央値 {d['median']:.1f}{unit} / "
            f"25% {d['p25']:.1f}{unit} / 75% {d['p75']:.1f}{unit} / 最大 {d['max']:.0f}{unit}")


def fmt_drop(s: pd.Series) -> str:
    d = describe(s, worst_is_min=True)
    if d.get("n", 0) == 0:
        return "標本なし"
    w = " ⚠️標本10件未満" if d["n"] < SMALL_SAMPLE else ""
    return (f"n={d['n']}{w} / 平均 {d['mean']:.1f}% / 中央値 {d['median']:.1f}% / "
            f"25% {d['p25']:.1f}% / 75% {d['p75']:.1f}% / 最も深い下落 {d['max']:.1f}%")


def fmt_pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.1f}%"


# --------------------------------------------------------------------------- 図
def make_plot(df: pd.DataFrame, path: Path, window: int, min_drop: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for f in ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "Hiragino Sans", "Yu Gothic"]:
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    c_all, c_drop = "#2a78d6", "#eb6834"

    tickers = list(df["ticker"].unique())
    fig, axes = plt.subplots(len(tickers), 2, figsize=(11, 3.4 * len(tickers)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.2, 1]}, facecolor=surf)
    bins = np.arange(0, window + 10, 10)
    for i, t in enumerate(tickers):
        d = df[df["ticker"] == t]
        a = d["days_to_bottom_trading"].dropna()
        b = d.loc[d["max_drop_pct"] <= min_drop, "days_to_bottom_trading"].dropna()
        ax, bx = axes[i]
        for x in (ax, bx):
            x.set_facecolor(surf)
            for sp in ("top", "right"):
                x.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                x.spines[sp].set_color(grid)
            x.tick_params(colors=ink2, labelsize=9)
            x.grid(axis="y" if x is ax else "x", color=grid, linewidth=0.8)
            x.set_axisbelow(True)
        ax.hist(a, bins=bins, color=c_all, edgecolor=surf, linewidth=2, label=f"全イベント (n={len(a)})")
        ax.hist(b, bins=bins, color=c_drop, edgecolor=surf, linewidth=2,
                label=f"最大下落 {min_drop:.0f}%以下 (n={len(b)})")
        ax.set_title(f"{t}：底までの営業日数", loc="left", color=ink, fontsize=11)
        ax.set_xlabel("起点からの営業日数", color=ink2, fontsize=9)
        ax.set_ylabel("イベント数", color=ink2, fontsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=ink2)
        data, labels = [], []
        for s, lab in [(a, "全イベント"), (b, f"{min_drop:.0f}%以下")]:
            if len(s):
                data.append(s.values)
                labels.append(f"{lab}\n(n={len(s)})")
        if data:
            bp = bx.boxplot(data, vert=False, tick_labels=labels, widths=0.5, patch_artist=True,
                            medianprops={"color": ink, "linewidth": 2},
                            whiskerprops={"color": ink2}, capprops={"color": ink2},
                            flierprops={"markerfacecolor": ink2, "markeredgecolor": surf, "markersize": 8})
            for patch, c in zip(bp["boxes"], [c_all, c_drop]):
                patch.set_facecolor(c)
                patch.set_edgecolor(surf)
        bx.set_xlim(0, window + 5)
        bx.set_title("箱ひげ図", loc="left", color=ink, fontsize=11)
        bx.set_xlabel("営業日数", color=ink2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


# --------------------------------------------------------------------------- summary.md
def write_summary(df: pd.DataFrame, all_df: pd.DataFrame, path: Path, args, sources: dict,
                  fetch_logs: dict, rec_src: str) -> None:
    L = []
    L.append("# イベント起点の「底までの日数」検証結果\n")
    L.append(f"- 実行日: {date.today().isoformat()}")
    L.append(f"- 観測期間: {args.window} 営業日 / 大幅下落の閾値: {args.min_drop:.1f}%")
    L.append(f"- 「短期で底」= 底まで {args.short_days} 営業日以内 / 「長期化」= {args.long_days} 営業日以上")
    L.append(f"- 景気後退期の判定: {rec_src}")
    L.append("")
    L.append("> **注意**: 以下は過去データの記述統計です。標本数が少なく、**統計的に有意とは言えません**。"
             "過去の傾向は**将来の値動きを保証しません**。投資判断の推奨ではありません。\n")
    L.append("## 定義\n")
    L.append("- 起点 = イベント日より前の最後の取引日の終値（起点はイベント日以前のデータだけで決定）。")
    L.append("- 選挙イベントは「選挙翌営業日」をイベント日とするため、起点は選挙当日の終値。")
    L.append(f"- 底 = イベント日以降 {args.window} 営業日以内の最安終値。日数は起点日からの営業日数・暦日数。")
    L.append("- 回復 = 底の後に終値が起点終値を初めて上回った日。期間内に無ければ「未回復」。")
    L.append("- 底が観測期間の最終日だった場合は「打ち切り」（期間を延ばせばさらに下がった可能性がある）。")
    L.append("- 観測期間が完全にそろわないイベント（直近のイベント等）とデータ欠損イベントは集計から除外し、下に列挙。")
    L.append("- 景気後退期（NBER）は事後に判定されるため、当時の投資家が知り得た情報ではない（事後の分類）。\n")

    L.append("## データ取得状況\n")
    for t in args.ticker:
        if t in sources:
            L.append(f"- **{t}**: {sources[t].source}（{sources[t].close.index[0].date()} 〜 "
                     f"{sources[t].close.index[-1].date()}、{len(sources[t].close)} 日）")
        else:
            L.append(f"- **{t}**: **取得できず**（{' / '.join(fetch_logs.get(t, []))}）")
    L.append("")

    for t in args.ticker:
        if t not in sources:
            continue
        d = df[df["ticker"] == t]
        excl = all_df[(all_df["ticker"] == t) & (all_df["status"] != "ok")]
        L.append(f"## {t}\n")
        L.append(f"集計対象 {len(d)} 件（除外 {len(excl)} 件）。\n")
        L.append("### 底までの日数（全イベント）\n")
        L.append(f"- 営業日: {fmt_stats(describe(d['days_to_bottom_trading']))}")
        L.append(f"- 暦日: {fmt_stats(describe(d['days_to_bottom_calendar']))}")
        L.append(f"- 最大下落率: {fmt_drop(d['max_drop_pct'])}")
        rec_n = (d["recovery"] == "回復").sum()
        unrec = (d["recovery"] == "未回復").sum()
        nodrop = (d["recovery"] == "下落なし").sum()
        L.append(f"- 回復状況: 回復 {rec_n} 件 / 未回復 {unrec} 件 / 起点を下回らず {nodrop} 件")
        L.append(f"- 回復までの営業日（回復したもののみ）: "
                 f"{fmt_stats(describe(d.loc[d['recovery'] == '回復', 'recovery_days_trading']))}")
        L.append(f"- 底が観測期間末尾（打ち切り）: {int(d['bottom_at_window_end'].sum())} 件\n")

        big = d[d["max_drop_pct"] <= args.min_drop]
        L.append(f"### 最大下落率が {args.min_drop:.1f}% 以下のイベントのみ\n")
        L.append(f"- 営業日: {fmt_stats(describe(big['days_to_bottom_trading']))}")
        L.append(f"- 暦日: {fmt_stats(describe(big['days_to_bottom_calendar']))}")
        L.append(f"- 最大下落率: {fmt_drop(big['max_drop_pct'])}")
        L.append(f"- 回復までの営業日（回復したもののみ）: "
                 f"{fmt_stats(describe(big.loc[big['recovery'] == '回復', 'recovery_days_trading']))}")
        L.append(f"- 未回復: {(big['recovery'] == '未回復').sum()} 件\n")

        L.append("### 短期で底を付けたケースと長期化したケースの比較\n")
        dd = d[d["max_drop_pct"] < 0]
        groups = [("短期", dd[dd["days_to_bottom_trading"] <= args.short_days]),
                  ("中間", dd[(dd["days_to_bottom_trading"] > args.short_days) &
                             (dd["days_to_bottom_trading"] < args.long_days)]),
                  ("長期化", dd[dd["days_to_bottom_trading"] >= args.long_days])]
        L.append("起点を一度でも下回ったイベントのみ。数値は中央値。\n")
        L.append("| 区分 | 件数 | 直前1か月リターン | 直前12か月リターン | 最大下落率 | 景気後退期の件数 |")
        L.append("|---|---|---|---|---|---|")
        for name, g in groups:
            n = len(g)
            rec_known = g["recession"].dropna() if "recession" in g else pd.Series(dtype=object)
            rec_txt = f"{int((rec_known == True).sum())}/{len(rec_known)}" if len(rec_known) else "不明"  # noqa: E712
            warn = " ⚠️" if n < SMALL_SAMPLE else ""
            L.append(f"| {name} | {n}{warn} | {fmt_pct(g['ret_1m_before'].median() if n else np.nan)} | "
                     f"{fmt_pct(g['ret_12m_before'].median() if n else np.nan)} | "
                     f"{fmt_pct(g['max_drop_pct'].median() if n else np.nan)} | {rec_txt} |")
        L.append("\n⚠️ = 標本10件未満。比較は参考程度で、差があっても偶然の範囲である可能性が高い。\n")

        L.append("### イベント別一覧\n")
        L.append("| イベント | 起点日 | 底の日 | 底まで(営業日/暦日) | 最大下落 | 回復(営業日) | 直前1M | 直前12M | 景気後退 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in d.iterrows():
            rec_d = (f"{int(r['recovery_days_trading'])}" if r["recovery"] == "回復"
                     else r["recovery"])
            rc = {True: "○", False: "", None: "不明"}.get(r["recession"], "不明")
            cens = "（打ち切り）" if r["bottom_at_window_end"] else ""
            L.append(f"| {r['label']} | {r['origin_date']} | {r['bottom_date']}{cens} | "
                     f"{int(r['days_to_bottom_trading'])}/{int(r['days_to_bottom_calendar'])} | "
                     f"{fmt_pct(r['max_drop_pct'])} | {rec_d} | {fmt_pct(r['ret_1m_before'])} | "
                     f"{fmt_pct(r['ret_12m_before'])} | {rc} |")
        if len(excl):
            L.append("\n**集計から除外したイベント**\n")
            for _, r in excl.iterrows():
                L.append(f"- {r['label']}（{r['event_date']}）: {r['status']}")
        gaps = d[d["data_gap_flag"] == True]  # noqa: E712
        if len(gaps):
            L.append(f"\n**データ間隔が{GAP_DAYS}暦日を超えた区間を含むイベント**（休場の可能性もあるが補完はしていない）\n")
            for _, r in gaps.iterrows():
                L.append(f"- {r['label']}: 最大 {r['max_gap_calendar_days']} 暦日")
        L.append("")

    L.append("## 留意点\n")
    L.append("- 標本数が少ないため、平均や中央値は少数のイベントに大きく左右される。統計的に有意な結論は導けない。")
    L.append("- イベント以外の要因（金融政策、景気、個別材料）が同時期の値動きに影響している。因果関係は示していない。")
    L.append("- 個別銘柄・ETFは上場以前のイベントを欠損として除外しているため、指数とは標本が異なる。")
    L.append("- 本結果は「過去にこうだった」という事実の記述であり、将来を保証するものではなく、投資助言でもない。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", nargs="+", default=["^GSPC"], help="例: ^GSPC ^NDX NVDA SOXX")
    ap.add_argument("--events", help="イベントCSV（列: date,label[,category]）。省略時は既定のイベント")
    ap.add_argument("--window", type=int, default=120, help="観測期間（営業日）")
    ap.add_argument("--min-drop", type=float, default=-3.0, help="大幅下落とみなす閾値（%%、負の値）")
    ap.add_argument("--short-days", type=int, default=10, help="短期で底とみなす営業日数の上限")
    ap.add_argument("--long-days", type=int, default=40, help="長期化とみなす営業日数の下限")
    ap.add_argument("--outdir", default="output", help="出力先ディレクトリ")
    ap.add_argument("--price-dir", help="ローカル価格CSVのディレクトリ（<ticker>.csv, 列: Date,Close）。"
                                        "あればネット取得より優先")
    args = ap.parse_args()
    if args.min_drop > 0:
        args.min_drop = -args.min_drop

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    events = load_events(args.events)
    start = (events["date"].min() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")

    rec, rec_src = load_recession(args.price_dir)
    print(f"景気後退期データ: {rec_src}")

    sources, logs, rows = {}, {}, []
    for t in args.ticker:
        ps, log = load_prices(t, start, args.price_dir)
        logs[t] = log
        for m in log:
            print(f"[{t}] {m}")
        if ps is None:
            print(f"[{t}] 全ソースで取得失敗。このティッカーは欠損として報告します。")
            for _, e in events.iterrows():
                rows.append({"ticker": t, "event_date": e["date"].date(), "label": e["label"],
                             "category": e["category"], "status": "欠損: 価格データ取得失敗"})
            continue
        sources[t] = ps
        print(f"[{t}] {ps.source}: {ps.close.index[0].date()} 〜 {ps.close.index[-1].date()}")
        for _, e in events.iterrows():
            r = analyze_event(ps.close, e["date"], args.window, rec)
            rows.append({"ticker": t, "event_date": e["date"].date(), "label": e["label"],
                         "category": e["category"], "source": ps.source, **r})

    all_df = pd.DataFrame(rows)
    cols = ["ticker", "event_date", "label", "category", "status", "source", "origin_date", "origin_close",
            "bottom_date", "bottom_close", "days_to_bottom_trading", "days_to_bottom_calendar", "max_drop_pct",
            "bottom_at_window_end", "recovery", "recovery_date", "recovery_days_trading",
            "recovery_days_calendar", "ret_1m_before", "ret_12m_before", "recession",
            "window_days_available", "window_complete", "max_gap_calendar_days", "data_gap_flag"]
    for c in cols:
        if c not in all_df.columns:
            all_df[c] = np.nan
    all_df = all_df[cols]
    all_df.round(4).to_csv(out / "results.csv", index=False, encoding="utf-8-sig")

    ok = all_df[all_df["status"] == "ok"].copy()
    if ok.empty:
        print("集計可能なイベントがありません。results.csv のみ出力しました。")
    else:
        make_plot(ok, out / "distribution.png", args.window, args.min_drop)
    write_summary(ok, all_df, out / "summary.md", args, sources, logs, rec_src)
    written = [p for p in ("results.csv", "distribution.png", "summary.md") if (out / p).exists()]
    print("出力: " + ", ".join(str(out / p) for p in written))


if __name__ == "__main__":
    main()
