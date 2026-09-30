"""Detect unusual card transactions in SWT_DEMO.FRAUD.TRANSACTIONS.

Four rules, all evaluated in Snowflake:
  amount    z-score of LN(amount) against the cardholder's own history
  geo       spend outside the cardholder's home region
  velocity  6 or more authorisations in the same hour
  travel    two distant countries inside 60 minutes
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from snowflake.snowpark.context import get_active_session

SOURCE = "SWT_DEMO.FRAUD.TRANSACTIONS"
HOME_REGION = ("FR", "GB", "DE", "ES", "IT", "NL", "IE", "PT", "BE", "LU", "AT")
RULE_WEIGHTS = {"amount spike": 40, "unusual geography": 25,
                "velocity burst": 20, "impossible travel": 30}
MAX_RISK = sum(RULE_WEIGHTS.values())

# Matches chartCategoricalColors in .streamlit/config.toml.
BRAND = ["#29B5E8", "#FF8B00", "#36B37E", "#6554C0", "#DE350B", "#11567F"]
PLOT_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=13, color="#11567F"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=48, b=8),
    title=dict(font=dict(size=16)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
)

st.set_page_config(page_title="Transaction anomalies", page_icon="🔍", layout="wide")


def styled(fig):
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(gridcolor="#e8f4f8", zeroline=False)
    fig.update_yaxes(gridcolor="#e8f4f8", zeroline=False)
    return fig


@st.cache_resource
def get_session():
    return get_active_session()


@st.cache_data(ttl=600, show_spinner="Scoring transactions in Snowflake...")
def load_flagged(days: int, z_threshold: float, velocity_threshold: int,
                 travel_minutes: int) -> pd.DataFrame:
    region = ", ".join(f"'{c}'" for c in HOME_REGION)
    return get_session().sql(f"""
WITH base AS (
    SELECT transaction_id, account_id, ts, amount::FLOAT AS amount,
           merchant_country, home_country
    FROM {SOURCE}
    WHERE ts >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
),
acct AS (
    SELECT account_id,
           AVG(LN(amount))            AS mu,
           NVL(STDDEV(LN(amount)), 0) AS sd,
           AVG(amount)                AS avg_amount
    FROM base
    GROUP BY account_id
),
scored AS (
    SELECT b.transaction_id, b.account_id, b.ts, b.amount,
           b.merchant_country, b.home_country, a.avg_amount,
           CASE WHEN a.sd > 0 THEN (LN(b.amount) - a.mu) / a.sd ELSE 0 END AS amount_z,
           COUNT(*) OVER (PARTITION BY b.account_id, DATE_TRUNC('hour', b.ts)) AS tx_same_hour,
           LAG(b.merchant_country) OVER (PARTITION BY b.account_id ORDER BY b.ts) AS prev_country,
           LAG(b.ts) OVER (PARTITION BY b.account_id ORDER BY b.ts) AS prev_ts
    FROM base b
    JOIN acct a ON a.account_id = b.account_id
),
flagged AS (
    SELECT s.*,
        (s.amount_z >= {z_threshold})                          AS f_amount,
        (s.merchant_country <> s.home_country
         AND s.merchant_country NOT IN ({region}))             AS f_geo,
        (s.tx_same_hour >= {velocity_threshold})               AS f_velocity,
        (s.prev_country IS NOT NULL
         AND s.prev_country <> s.merchant_country
         AND DATEDIFF(minute, s.prev_ts, s.ts) <= {travel_minutes}
         AND (s.merchant_country NOT IN ({region})
              OR s.prev_country NOT IN ({region})))            AS f_travel
    FROM scored s
)
SELECT transaction_id, account_id, ts, amount, merchant_country, home_country,
       ROUND(amount_z, 2)::FLOAT   AS amount_z,
       ROUND(avg_amount, 2)::FLOAT AS account_avg_amount,
       tx_same_hour,
       DATEDIFF(minute, prev_ts, ts) AS minutes_since_prev,
       f_amount, f_geo, f_velocity, f_travel,
       IFF(f_amount, 40, 0) + IFF(f_geo, 25, 0)
     + IFF(f_velocity, 20, 0) + IFF(f_travel, 30, 0) AS risk_score,
       ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(
           IFF(f_amount,   'amount spike',      NULL),
           IFF(f_geo,      'unusual geography', NULL),
           IFF(f_velocity, 'velocity burst',    NULL),
           IFF(f_travel,   'impossible travel', NULL)), ', ') AS reasons
FROM flagged
WHERE f_amount OR f_geo OR f_velocity OR f_travel
ORDER BY risk_score DESC, amount DESC
""").to_pandas()


@st.cache_data(ttl=600)
def load_volume(days: int) -> pd.DataFrame:
    return get_session().sql(f"""
        SELECT DATE_TRUNC('day', ts)::DATE AS day,
               COUNT(*)          AS tx_count,
               SUM(amount)::FLOAT AS tx_value
        FROM {SOURCE}
        WHERE ts >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
        GROUP BY day ORDER BY day
    """).to_pandas()


@st.cache_data(ttl=600)
def load_account_history(account_id: str, days: int) -> pd.DataFrame:
    return get_session().sql(f"""
        SELECT transaction_id, ts, amount::FLOAT AS amount, merchant_country, home_country
        FROM {SOURCE}
        WHERE account_id = ?
          AND ts >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
        ORDER BY ts
    """, params=[account_id]).to_pandas()


# --- Header and controls ----------------------------------------------------

st.title("Unusual transaction detection")
st.caption(f"Four rules evaluated in Snowflake against `{SOURCE}`")

with st.sidebar:
    st.subheader("Detection settings")
    days = st.slider("Lookback", 7, 90, 90, step=7, format="%d days")
    z_threshold = st.slider(
        "Amount z-score", 2.0, 6.0, 4.0, step=0.5,
        help="Standard deviations above the cardholder's own log-amount mean.")
    velocity_threshold = st.slider("Authorisations per hour", 3, 12, 6)
    travel_minutes = st.slider("Travel window", 15, 180, 60, step=15, format="%d min")
    st.divider()
    st.caption("Rule weights")
    for rule, weight in RULE_WEIGHTS.items():
        st.progress(weight / MAX_RISK, text=f"{rule} · {weight}")

flagged = load_flagged(days, z_threshold, velocity_threshold, travel_minutes)
volume = load_volume(days)
total_tx = int(volume["TX_COUNT"].sum())

with st.popover("Filters", use_container_width=False):
    min_risk = st.select_slider("Minimum risk score", options=[0, 25, 40, 55, 70, 95], value=0)
    all_countries = sorted(flagged["MERCHANT_COUNTRY"].dropna().unique().tolist())
    countries = st.multiselect("Merchant country", all_countries, default=all_countries)
    rules = st.multiselect("Rule", list(RULE_WEIGHTS), default=list(RULE_WEIGHTS))

view = flagged[flagged["RISK_SCORE"] >= min_risk]
if countries:
    view = view[view["MERCHANT_COUNTRY"].isin(countries)]
if rules and len(rules) < len(RULE_WEIGHTS):
    view = view[view["REASONS"].apply(lambda r: any(rule in r for rule in rules))]

# --- Headline numbers -------------------------------------------------------

flag_rate = (len(view) / total_tx * 100) if total_tx else 0.0
high_risk = int((view["RISK_SCORE"] >= 55).sum())

c1, c2, c3, c4 = st.columns(4, border=True)
c1.metric("Transactions scored", f"{total_tx:,}")
c2.metric("Flagged", f"{len(view):,}", f"{flag_rate:.2f}% of volume", delta_color="off")
c3.metric("High risk", f"{high_risk:,}", "score 55 or above", delta_color="off")
c4.metric("Value flagged", f"{view['AMOUNT'].sum():,.0f}")

tab_queue, tab_patterns, tab_account = st.tabs(
    ["Review queue", "Patterns", "Account drill-down"])

with tab_queue:
    if view.empty:
        st.info("No transactions match the current filters.")
    else:
        st.dataframe(
            view[["TRANSACTION_ID", "ACCOUNT_ID", "TS", "AMOUNT", "MERCHANT_COUNTRY",
                  "HOME_COUNTRY", "RISK_SCORE", "REASONS", "AMOUNT_Z",
                  "ACCOUNT_AVG_AMOUNT", "TX_SAME_HOUR", "MINUTES_SINCE_PREV"]],
            use_container_width=True, hide_index=True, height=440,
            column_config={
                "TRANSACTION_ID": "Transaction",
                "ACCOUNT_ID": "Account",
                "TS": st.column_config.DatetimeColumn("Timestamp", format="D MMM YYYY, HH:mm"),
                "AMOUNT": st.column_config.NumberColumn("Amount", format="%.2f"),
                "MERCHANT_COUNTRY": "Merchant",
                "HOME_COUNTRY": "Home",
                "RISK_SCORE": st.column_config.ProgressColumn(
                    "Risk", min_value=0, max_value=MAX_RISK, format="%d"),
                "REASONS": "Why it fired",
                "AMOUNT_Z": st.column_config.NumberColumn("z-score", format="%.2f"),
                "ACCOUNT_AVG_AMOUNT": st.column_config.NumberColumn("Account avg", format="%.2f"),
                "TX_SAME_HOUR": st.column_config.NumberColumn("Tx/hour"),
                "MINUTES_SINCE_PREV": st.column_config.NumberColumn("Min since prev"),
            },
        )
        st.download_button("Download review queue (CSV)",
                           view.to_csv(index=False).encode("utf-8"),
                           file_name="flagged_transactions.csv", mime="text/csv")

with tab_patterns:
    if view.empty:
        st.info("Nothing to chart.")
    else:
        left, right = st.columns(2)

        by_rule = pd.DataFrame({
            "rule": list(RULE_WEIGHTS),
            "transactions": [int(view[c].sum()) for c in
                             ("F_AMOUNT", "F_GEO", "F_VELOCITY", "F_TRAVEL")],
        }).sort_values("transactions")
        left.plotly_chart(styled(px.bar(
            by_rule, x="transactions", y="rule", orientation="h", text="transactions",
            title="Flags by rule", color_discrete_sequence=BRAND)), use_container_width=True)

        by_country = (view.groupby("MERCHANT_COUNTRY", as_index=False)
                      .agg(transactions=("TRANSACTION_ID", "count"))
                      .sort_values("transactions", ascending=False).head(12))
        right.plotly_chart(styled(px.bar(
            by_country, x="MERCHANT_COUNTRY", y="transactions", text="transactions",
            title="Flagged volume by merchant country",
            color_discrete_sequence=BRAND)), use_container_width=True)

        st.plotly_chart(styled(px.scatter(
            view, x="TS", y="AMOUNT", color="REASONS", size="RISK_SCORE",
            size_max=22, log_y=True, hover_data=["TRANSACTION_ID", "ACCOUNT_ID"],
            title="Flagged transactions over time (log amount)",
            color_discrete_sequence=BRAND)), use_container_width=True)

        by_day = (view.assign(DAY=pd.to_datetime(view["TS"]).dt.date)
                  .groupby("DAY", as_index=False).agg(flagged=("TRANSACTION_ID", "count")))
        volume["DAY"] = pd.to_datetime(volume["DAY"]).dt.date
        trend = volume.merge(by_day, on="DAY", how="left").fillna({"flagged": 0})
        st.plotly_chart(styled(px.line(
            trend, x="DAY", y=["TX_COUNT", "flagged"], markers=False,
            title="Daily volume against flags",
            color_discrete_sequence=BRAND)), use_container_width=True)

with tab_account:
    if view.empty:
        st.info("No accounts to inspect.")
    else:
        ranked = (view.groupby("ACCOUNT_ID", as_index=False)
                  .agg(flags=("TRANSACTION_ID", "count"), risk=("RISK_SCORE", "max"))
                  .sort_values(["risk", "flags"], ascending=False))
        account_id = st.selectbox(
            "Account", ranked["ACCOUNT_ID"].tolist(),
            format_func=lambda a: (
                f"{a} · {int(ranked.loc[ranked.ACCOUNT_ID == a, 'flags'].iloc[0])} flags "
                f"· max risk {int(ranked.loc[ranked.ACCOUNT_ID == a, 'risk'].iloc[0])}"))

        history = load_account_history(account_id, days)
        account_flags = view[view["ACCOUNT_ID"] == account_id]
        flagged_ids = set(account_flags["TRANSACTION_ID"])
        history["status"] = history["TRANSACTION_ID"].apply(
            lambda t: "Flagged" if t in flagged_ids else "Normal")

        m1, m2, m3 = st.columns(3, border=True)
        m1.metric("Transactions", f"{len(history):,}")
        m2.metric("Flagged", f"{len(account_flags):,}")
        m3.metric("Median amount", f"{history['AMOUNT'].median():,.2f}")

        st.plotly_chart(styled(px.scatter(
            history, x="TS", y="AMOUNT", color="status", log_y=True,
            color_discrete_map={"Flagged": "#DE350B", "Normal": "#B8CDD9"},
            hover_data=["TRANSACTION_ID", "MERCHANT_COUNTRY"],
            title=f"{account_id} spend history")), use_container_width=True)

        st.subheader("Flagged for this account")
        st.dataframe(
            account_flags[["TRANSACTION_ID", "TS", "AMOUNT", "MERCHANT_COUNTRY",
                           "RISK_SCORE", "REASONS"]],
            use_container_width=True, hide_index=True,
            column_config={
                "TS": st.column_config.DatetimeColumn("Timestamp", format="D MMM YYYY, HH:mm"),
                "AMOUNT": st.column_config.NumberColumn("Amount", format="%.2f"),
                "RISK_SCORE": st.column_config.ProgressColumn(
                    "Risk", min_value=0, max_value=MAX_RISK, format="%d"),
            })
