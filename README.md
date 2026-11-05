# イベント起点の「底までの日数」検証

重大イベント日（選挙・地政学ショック・FOMC・決算など）を起点に、指数や銘柄が何営業日後に底を付け、何日で起点の水準に戻ったかを過去データで集計するスクリプトです。

```bash
pip install -r requirements.txt
python event_bottom_analysis.py --ticker ^GSPC ^NDX NVDA SOXX
python event_bottom_analysis.py --ticker ^GSPC --events events_example.csv --window 60 --min-drop -5
python tests/test_logic.py   # 計算ロジックの単体テスト（ネット不要）
```

出力（`--outdir`、既定は `output/`）: `results.csv`（イベントごとの指標）、`distribution.png`（底までの日数のヒストグラムと箱ひげ図）、`summary.md`（日本語の要約）。

- データ: yfinance の調整後終値 → 失敗時は stooq → FRED（指数のみ）。取得できない期間は欠損として報告し、補完しない。
- 既定イベント: 1980年以降の米大統領選・中間選挙の翌営業日（選挙日は11月第1月曜の翌日として計算）と、地政学ショック4件（1990年のクウェート侵攻、9/11、2003年のイラク侵攻、2022年のウクライナ侵攻）。
- 起点 = イベント日より前の最後の終値。観測期間が足りない直近のイベントは集計から除外する。
- 景気後退期は FRED の USREC（NBER の事後判定）。当時の投資家が知り得た情報ではない。
- `--price-dir` に `<ticker>.csv`（列: Date,Close）や `USREC.csv` を置くと、ネットからの取得より優先して使う。

結果は過去の記述統計で、統計的に有意とは言えず、将来を保証するものでも投資助言でもありません。
