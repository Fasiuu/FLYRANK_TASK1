"""Shared feature + label builder for the capstone (Growth / Recovery / Momentum lane).

One row = one content item at one decision date T.
  features : built ONLY from days in (T-90, T]          -> knowable at T
  label    : built ONLY from days in (T, T+30]           -> the future month
  declined = the page's next-30-day impressions, relative to its own prior 60-day monthly average,
             fall more than 20% BELOW what its own client (site) did over the same month.
             (page_ratio < 0.8 x client_ratio). Site-wide swings - algorithm updates, seasonality,
             tracking changes - move every page of a site together and are not something a page
             refresh can fix, so they are removed from the label. `declined_abs` keeps the plain
             >20%-drop version for comparison.

Everything heavy runs inside DuckDB against the hosted Parquet release; pandas only ever sees
the per-item aggregate (tens of thousands of rows, not millions).
"""
import datetime as dt
import os

import duckdb
import pandas as pd

REL = "hf://datasets/FlyRank/internship-warehouse"
SEED = 42

# Decision dates. T1/T2 are the development snapshots; T3's label month (June 2026, the last
# month of the panel) is the sealed test month and is only scored once, in w06.
DEV_DATES = ["2026-03-31", "2026-04-30"]
SEALED_DATE = "2026-05-31"
ALL_DATES = DEV_DATES + [SEALED_DATE]
DEPLOY_DATE = "2026-06-30"   # last day of the panel: features only, the label month is the real future
PANEL_END = "2026-06-30"

MIN_IMP_60D = 200          # population floor: >= 200 impressions over (T-60, T]
MIN_AGE_DAYS = 90          # page must exist for the whole 90-day feature window
DECLINE_RATIO = 0.8        # >20% drop vs the prior 60-day monthly average

FEATURES = [
    "log_imp_last30", "log_imp_prev30", "log_imp_prev60_90",
    "imp_momentum", "imp_momentum_long",
    "log_clicks_last30", "ctr_last30",
    "pos_last30", "pos_change", "pos_volatility",
    "days_with_imp_last30", "spike_share", "weekend_share",
    "content_age_days", "log_word_count", "has_word_count",
    "log_search_volume", "has_keyword_data",
]


def get_token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
        except Exception:
            tok = None
    if not tok:
        import getpass
        tok = getpass.getpass("Hugging Face READ token (hf_...): ")
    return tok


def connect():
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{get_token()}')")
    return con


def month_paths(start, end):
    """Explicit list of month partitions covering [start, end] (hf:// has no brace globs)."""
    s = dt.date.fromisoformat(start).replace(day=1)
    e = dt.date.fromisoformat(end)
    out = []
    while s <= e:
        out.append(f"'{REL}/fact_content_daily_performance/month={s:%Y-%m}/*.parquet'")
        s = (s.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return "[" + ", ".join(out) + "]"


def _panel_sql(dates, paths):
    dates_sql = ", ".join(f"(DATE '{T}')" for T in dates)
    return f"""
    WITH t(decision_date) AS (VALUES {dates_sql}),
    raw AS (
        SELECT client_hash_id, content_hash_id, report_date,
               COALESCE(gsc_impressions, 0) AS imp,
               COALESCE(gsc_clicks, 0)      AS clk,
               gsc_avg_position             AS pos
        FROM read_parquet({paths})
    ),
    d AS (   -- one scan of the fact table; each day is attached to every decision date it serves
        SELECT t.decision_date, r.*, (t.decision_date - r.report_date) AS age   -- 0 = decision day
        FROM raw r JOIN t
          ON r.report_date >  t.decision_date - INTERVAL 90 DAY
         AND r.report_date <= t.decision_date + INTERVAL 30 DAY
    ),
    agg AS (
        SELECT decision_date, client_hash_id, content_hash_id,
            -- feature window (T-90, T]
            SUM(imp) FILTER (WHERE age BETWEEN 0 AND 29)  AS imp_last30,
            SUM(imp) FILTER (WHERE age BETWEEN 30 AND 59) AS imp_prev30,
            SUM(imp) FILTER (WHERE age BETWEEN 60 AND 89) AS imp_prev60_90,
            SUM(clk) FILTER (WHERE age BETWEEN 0 AND 29)  AS clicks_last30,
            SUM(pos * imp) FILTER (WHERE age BETWEEN 0 AND 29 AND imp > 0)
              / NULLIF(SUM(imp) FILTER (WHERE age BETWEEN 0 AND 29 AND imp > 0), 0) AS pos_last30,
            SUM(pos * imp) FILTER (WHERE age BETWEEN 30 AND 59 AND imp > 0)
              / NULLIF(SUM(imp) FILTER (WHERE age BETWEEN 30 AND 59 AND imp > 0), 0) AS pos_prev30,
            STDDEV_SAMP(pos) FILTER (WHERE age BETWEEN 0 AND 29 AND imp > 0) AS pos_volatility,
            COUNT(*) FILTER (WHERE age BETWEEN 0 AND 29 AND imp > 0) AS days_with_imp_last30,
            MAX(imp) FILTER (WHERE age BETWEEN 0 AND 29) AS max_day_imp_last30,
            SUM(imp) FILTER (WHERE age BETWEEN 0 AND 29 AND dayofweek(report_date) IN (0, 6)) AS weekend_imp_last30,
            -- label window (T, T+30]   (age is negative there)
            SUM(imp) FILTER (WHERE age BETWEEN -30 AND -1) AS imp_next30,
            COUNT(*) FILTER (WHERE age BETWEEN -30 AND -1) AS rows_next30
        FROM d GROUP BY 1, 2, 3
    )
    SELECT * FROM agg
    WHERE COALESCE(imp_last30, 0) + COALESCE(imp_prev30, 0) >= {MIN_IMP_60D}
    """


def build_panel(con, dates=ALL_DATES):
    """Per-item feature + label rows for each decision date, from ONE scan of the needed months."""
    lo = (min(dt.date.fromisoformat(T) for T in dates) - dt.timedelta(days=90)).isoformat()
    hi = min(max(dt.date.fromisoformat(T) for T in dates) + dt.timedelta(days=30),
             dt.date.fromisoformat(PANEL_END)).isoformat()
    panel = con.sql(_panel_sql(dates, month_paths(lo, hi))).df()
    return add_context(con, panel)


def add_context(con, panel):
    content = con.sql(f"""
        SELECT content_hash_id,
               ANY_VALUE(content_created_date) AS content_created_date,
               ANY_VALUE(content_updated_date) AS content_updated_date,
               ANY_VALUE(word_count)           AS word_count,
               ANY_VALUE(search_volume)        AS search_volume,
               ANY_VALUE(content_type)         AS content_type
        FROM read_parquet('{REL}/dim_content.parquet') GROUP BY 1""").df()
    clients = con.sql(f"""
        SELECT client_hash_id, gsc_data_start FROM read_parquet('{REL}/dim_clients.parquet')""").df()
    panel = panel.merge(content, on="content_hash_id", how="left").merge(clients, on="client_hash_id", how="left")
    return engineer(panel)


def engineer(df):
    import numpy as np
    df = df.copy()
    T = pd.to_datetime(df["decision_date"])
    for c in ["imp_last30", "imp_prev30", "imp_prev60_90", "clicks_last30", "imp_next30",
              "weekend_imp_last30", "max_day_imp_last30", "days_with_imp_last30", "rows_next30"]:
        df[c] = df[c].fillna(0)
    # history guard: the client's GSC feed must cover the whole 90-day feature window
    df["full_history"] = pd.to_datetime(df["gsc_data_start"]) <= T - pd.Timedelta(days=89)
    df["log_imp_last30"] = np.log1p(df["imp_last30"])
    df["log_imp_prev30"] = np.log1p(df["imp_prev30"])
    df["log_imp_prev60_90"] = np.log1p(df["imp_prev60_90"])
    df["log_clicks_last30"] = np.log1p(df["clicks_last30"])
    df["imp_momentum"] = np.log1p(df["imp_last30"]) - np.log1p(df["imp_prev30"])
    df["imp_momentum_long"] = np.log1p(df["imp_last30"]) - np.log1p(df["imp_prev60_90"])
    df["ctr_last30"] = np.where(df["imp_last30"] > 0, 100 * df["clicks_last30"] / df["imp_last30"].clip(lower=1), 0.0)
    df["pos_change"] = df["pos_last30"] - df["pos_prev30"]        # > 0 = slipping (bigger number = worse)
    df["spike_share"] = np.where(df["imp_last30"] > 0, df["max_day_imp_last30"] / df["imp_last30"].clip(lower=1), 0.0)
    df["weekend_share"] = np.where(df["imp_last30"] > 0, df["weekend_imp_last30"] / df["imp_last30"].clip(lower=1), 0.0)
    created = pd.to_datetime(df["content_created_date"], errors="coerce")
    df["content_age_days"] = (T - created).dt.days
    df.loc[df["content_age_days"] < 0, "content_age_days"] = np.nan   # created after T: not knowable
    df["log_word_count"] = np.log1p(df["word_count"].fillna(0))
    df["has_word_count"] = df["word_count"].notna().astype(int)
    df["log_search_volume"] = np.log1p(df["search_volume"].fillna(0))
    df["has_keyword_data"] = df["search_volume"].notna().astype(int)
    # EXCLUDED from features (kept only for the leakage hunt): the CURRENT update date is an
    # export-time value - a page edited after T carries information from after the decision point.
    df["updated_after_T"] = (pd.to_datetime(df["content_updated_date"], errors="coerce") > T).astype(int)
    # label: observed outcome in the following 30 days
    df["baseline_monthly_imp"] = (df["imp_last30"] + df["imp_prev30"]) / 2
    df["declined_abs"] = (df["imp_next30"] < DECLINE_RATIO * df["baseline_monthly_imp"]).astype(int)
    df["vanished_next30"] = (df["rows_next30"] == 0).astype(int)
    beyond = T + pd.Timedelta(days=30) > pd.Timestamp(PANEL_END)
    df.loc[beyond, ["declined_abs", "imp_next30", "vanished_next30"]] = np.nan
    return df


def model_frame(panel):
    """Rows used for modelling: full feature history, page old enough, every feature present,
    plus the site-relative label."""
    import numpy as np
    m = panel[panel["full_history"]].copy()
    m = m.dropna(subset=["pos_last30", "content_age_days"])
    m = m[m["content_age_days"] >= MIN_AGE_DAYS]
    m["pos_change"] = m["pos_change"].fillna(0)        # no position in the prior month -> no change measurable
    m["pos_volatility"] = m["pos_volatility"].fillna(0)  # a single visible day has no spread
    g = m.groupby(["decision_date", "client_hash_id"])
    m["client_ratio"] = g["imp_next30"].transform("sum") / g["baseline_monthly_imp"].transform("sum")
    m["page_ratio"] = m["imp_next30"] / m["baseline_monthly_imp"].clip(lower=1)
    m["declined"] = (m["page_ratio"] < DECLINE_RATIO * m["client_ratio"]).astype(float)
    m.loc[m["imp_next30"].isna(), ["declined", "client_ratio", "page_ratio"]] = np.nan
    return m


def client_fold(client_hash_id, k=4):
    """Fixed, data-independent fold id per client (hash of the pseudonymous id)."""
    import hashlib
    return int(hashlib.md5(f"{SEED}:{client_hash_id}".encode()).hexdigest(), 16) % k


def grouped_forward_scores(df, train_dates, test_date, make_model, features, k=4):
    """Client-grouped AND time-forward out-of-fold scores: for each fold of clients, train on the
    OTHER clients' rows at train_dates and score this fold's rows at test_date. Every test row is
    scored by a model that never saw its client, nor its outcome month."""
    import numpy as np
    test = df[df["decision_date"] == test_date].copy()
    test["fold"] = test["client_hash_id"].map(lambda c: client_fold(c, k))
    test["score"] = np.nan
    tr_all = df[df["decision_date"].isin(train_dates)]
    tr_fold = tr_all["client_hash_id"].map(lambda c: client_fold(c, k))
    for f in range(k):
        te_idx = test.index[test["fold"] == f]
        if len(te_idx) == 0:
            continue
        tr = tr_all[tr_fold != f]
        mdl = make_model().fit(tr[features], tr["declined"].astype(int))
        test.loc[te_idx, "score"] = mdl.predict_proba(test.loc[te_idx, features])[:, 1]
    return test


def is_holdout_client(client_hash_id):
    """Fixed, data-independent client split: ~1 in 4 clients is held out, decided by a hash of
    the pseudonymous id. Same answer in every notebook, whatever rows a notebook loaded."""
    import hashlib
    return int(hashlib.md5(f"{SEED}:{client_hash_id}".encode()).hexdigest(), 16) % 4 == 0


def client_split(df):
    clients = sorted(df["client_hash_id"].unique())
    test = {c for c in clients if is_holdout_client(c)}
    return set(clients) - test, test


def precision_at_k(scores, labels, k):
    import numpy as np
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return float(np.asarray(labels)[order[:k]].mean())
