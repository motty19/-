import sys, numpy as np, pandas as pd
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from event_bottom_analysis import analyze_event, election_days
idx = pd.bdate_range("2020-01-01", periods=400)
v = np.full(400, 100.0)
v[:300] = np.linspace(80, 100, 300)          # rising until origin
E = idx[300]                                   # event date
# origin = idx[299] close 100. drop to 90 at idx[304] (5th day), recover >100 at idx[310]
v[300:305] = [98, 96, 94, 92, 90]; v[305:310] = [92, 95, 97, 99, 100]; v[310:] = 101
px = pd.Series(v, idx)
r = analyze_event(px, E, 60, None)
assert r["origin_date"] == idx[299].date() and r["origin_close"] == 100
assert r["days_to_bottom_trading"] == 5 and r["bottom_close"] == 90
assert abs(r["max_drop_pct"] + 10) < 1e-9
assert r["recovery"] == "回復" and r["recovery_days_trading"] == 11  # 100 is not > 100; 101 at idx[310]
assert abs(r["ret_1m_before"] - (100 / v[299-21] - 1) * 100) < 1e-9
# Lookahead: changing future prices must not change origin / pre-returns
px2 = px.copy(); px2.iloc[300:] *= 0.5
r2 = analyze_event(px2, E, 60, None)
assert r2["origin_close"] == 100 and r2["ret_12m_before"] == r["ret_12m_before"]
# Unrecovered + censored bottom
px3 = px.copy(); px3.iloc[300:] = np.linspace(99, 70, 100)
r3 = analyze_event(px3, E, 60, None)
assert r3["recovery"] == "未回復" and r3["bottom_at_window_end"] and r3["days_to_bottom_trading"] == 60
# Incomplete window (data ends)
r4 = analyze_event(px, idx[380], 60, None)
assert r4["status"].startswith("不完全") and r4["window_days_available"] == 20
# Event before data -> missing
assert analyze_event(px, pd.Timestamp("2019-01-01"), 60, None)["status"].startswith("欠損")
# Event on weekend: origin = Friday close
r5 = analyze_event(px, pd.Timestamp("2021-02-13"), 10, None)
assert pd.Timestamp(r5["origin_date"]).weekday() == 4
# Election dates computed: 2020-11-03, 2024-11-05, 2022-11-08
d = dict((l, x) for x, l in election_days(2020, 2024))
assert str(d["2020 大統領選"]) == "2020-11-03" and str(d["2024 大統領選"]) == "2024-11-05" and str(d["2022 中間選挙"]) == "2022-11-08"
print("all logic tests passed")
