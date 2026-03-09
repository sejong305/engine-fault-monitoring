import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(page_title="Engine QC Monitoring Dashboard", layout="wide")
EPS = 1e-9

CHEVROLET_SAIL_BASE64 = """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"""


# =========================================================
# 스타일
# =========================================================
st.markdown("""
<style>
:root{
    --bg:#0f1722;
    --panel:#161f2b;
    --panel-2:#1c2633;
    --border:#2a3646;
    --text:#e8edf3;
    --muted:#9aa8b8;
    --green:#2fbf71;
    --red:#e05a5a;
    --amber:#d6a74f;
    --blue:#5b8def;
    --cyan:#4fb4d8;
}

html, body, [class*="css"]  {
    font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0b1220 0%, #101926 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 96rem;
}

section[data-testid="stSidebar"] {
    background: #121a25;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

div[data-testid="stFileUploader"] section {
    background: #17212d;
    border: 1px solid var(--border);
    border-radius: 12px;
}

.stSlider [data-baseweb="slider"] {
    margin-top: 0.25rem;
}

h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

.small-note {
    color: var(--muted);
    font-size: 12px;
}

.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 18px 16px 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.card-title {
    font-size: 13px;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: .04em;
}

.metric-big {
    font-size: 32px;
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
}

.metric-sub {
    font-size: 13px;
    color: var(--muted);
    margin-top: 8px;
}

.kpi-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    min-height: 116px;
}

.kpi-label {
    font-size: 13px;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: .04em;
}

.kpi-value {
    font-size: 34px;
    font-weight: 800;
    color: var(--text);
}

.badge {
    display:inline-block;
    padding:6px 12px;
    border-radius:999px;
    font-weight:700;
    font-size:13px;
    letter-spacing:.01em;
}

.badge-green { background: rgba(47,191,113,.14); color: #71d69a; border:1px solid rgba(47,191,113,.3); }
.badge-red { background: rgba(224,90,90,.14); color: #f08a8a; border:1px solid rgba(224,90,90,.3); }
.badge-blue { background: rgba(91,141,239,.14); color: #86aef6; border:1px solid rgba(91,141,239,.3); }
.badge-amber { background: rgba(214,167,79,.14); color: #e0bf7a; border:1px solid rgba(214,167,79,.3); }

.panel-title {
    font-size: 18px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 12px;
}

.divider-line {
    height: 1px;
    background: var(--border);
    margin: 14px 0 6px 0;
}

.hero-wrap {
    background: linear-gradient(135deg, #151d29 0%, #1a2430 55%, #18212c 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 16px 38px rgba(0,0,0,.22);
}

.hero-caption {
    color: #b8c4d1;
    font-size: 12px;
    letter-spacing: .08em;
    text-transform: uppercase;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

[data-testid="stMetric"] {
    background: transparent;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    background: transparent;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 데이터 처리 함수
# =========================================================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Fault" in df.columns:
        df["Fault_3"] = df["Fault"].replace({3: 2}).astype(int)

    if {"O2", "MAP"}.issubset(df.columns):
        df["O2_MAP_Index"] = df["O2"] / (df["MAP"] + EPS)

    if {"Power", "Consumption L/H"}.issubset(df.columns):
        df["Fuel_Efficiency"] = df["Power"] / (df["Consumption L/H"] + EPS)

    if {"AFR", "Lambda"}.issubset(df.columns):
        df["AFR_Deviation"] = df["AFR"] - 14.7 * df["Lambda"]

    if {"CO2", "CO", "HC"}.issubset(df.columns):
        df["Combustion_Quality"] = df["CO2"] / (df["CO"] + df["HC"] + EPS)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"Fault", "Fault_3"}
    return [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]


def fault_type_label(v: int) -> str:
    if int(v) == 0:
        return "-"
    if int(v) == 1:
        return "Rich mixture"
    return "Lean mixture or Low voltage"


def status_label(v: int) -> str:
    return "정상" if int(v) == 0 else "불량"


def get_md_18_columns(df: pd.DataFrame) -> list[str]:
    md_cols = [
        "MAP",
        "TPS",
        "Force",
        "Power",
        "RPM",
        "Consumption L/H",
        "Consumption L/100KM",
        "Speed",
        "CO",
        "HC",
        "CO2",
        "O2",
        "Lambda",
        "AFR",
        "O2_MAP_Index",
        "Fuel_Efficiency",
        "AFR_Deviation",
        "Combustion_Quality"
    ]
    return [c for c in md_cols if c in df.columns]


def compute_health_distance(df: pd.DataFrame, normal_mask: pd.Series) -> pd.Series:
    md_cols = get_md_18_columns(df)

    if len(md_cols) < 2:
        return pd.Series(np.nan, index=df.index)

    X = df[md_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    X_normal = X.loc[normal_mask].copy()

    if len(X_normal) < 10:
        return pd.Series(np.nan, index=df.index)

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_scaled = scaler.transform(X)

    lw = LedoitWolf()
    lw.fit(X_normal_scaled)

    mu = lw.location_
    cov = lw.covariance_
    inv_cov = np.linalg.inv(cov)

    diff = X_scaled - mu
    md_sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    md = np.sqrt(np.maximum(md_sq, 0))

    return pd.Series(md, index=df.index)


def build_baseline(df: pd.DataFrame, normal_mask: pd.Series, cols: list[str]) -> dict:
    base = df.loc[normal_mask, cols].copy()
    mean = base.mean(numeric_only=True)
    std = base.std(numeric_only=True).replace(0, np.nan) + EPS
    return {"mean": mean.to_dict(), "std": std.to_dict(), "n": int(normal_mask.sum())}


def make_sensor_current_table(row: pd.Series, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        val = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        rows.append({
            "변수": c,
            "현재값": val
        })

    t = pd.DataFrame(rows)
    if t.empty:
        return t

    return t.reset_index(drop=True)


def get_fault_insight_bundle(fault_text: str) -> dict:
    if fault_text == "Rich mixture":
        return {
            "summary": "연료가 상대적으로 과다한 상태로 판단되며, 연소 효율 저하와 촉매 부담 증가 가능성이 있습니다.",
            "major_factors": [
                "연료 분사량 과다 또는 인젝터 제어 이상",
                "흡기 유량 저하(에어필터 막힘, 흡기 계통 저항 증가)",
                "산소 센서 피드백 이상 또는 촉매 전단 센서 편차",
                "점화 불완전으로 인한 미연소 연료 증가"
            ],
            "checklist": [
                "산소 센서(O2 Sensor)",
                "인젝터 및 연료압 조절 상태",
                "에어필터 / 흡기 덕트 막힘 여부",
                "점화 플러그 / 점화 코일 상태",
                "촉매 변환기 과열 또는 열화 여부"
            ]
        }
    elif fault_text == "Lean mixture or Low voltage":
        return {
            "summary": "희박 연소 또는 전압 불안정 가능성이 함께 관찰되어, 연료 공급과 전원 계통을 동시에 확인할 필요가 있습니다.",
            "major_factors": [
                "흡기 누설 또는 진공 라인 손상으로 인한 희박 연소",
                "연료 공급 부족(연료 펌프, 필터, 인젝터 유량 저하)",
                "배터리/알터네이터 성능 저하로 인한 센서·점화 계통 전압 불안정",
                "점화 에너지 부족으로 인한 연소 편차 확대"
            ],
            "checklist": [
                "흡기 호스 / 진공 라인 누설 여부",
                "연료 펌프 / 연료 필터 / 인젝터 유량",
                "배터리 전압 및 충전 상태",
                "알터네이터 출력 안정성",
                "점화 코일 / 점화 플러그 / 주요 배선 커넥터"
            ]
        }
    else:
        return {
            "summary": "현재 엔진 센서 데이터는 정상 범위에 가까우나, 주요 배출·연소 지표가 기준치 이상입니다.",
            "major_factors": [
                "현재 정상 운행 가능",
                "엔진 센서 데이터 간 상관관계가 정상 패턴을 유지함",
                "차량 연식(12년식)에 따른 배출 성능 저하 의심"
            ],
            "checklist": [
                "흡기 필터 및 흡기 계통 오염 상태 점검",
                "산소 센서(O2 Sensor) 노후 및 신호 이상 여부 확인",
                "산소 센서 및 배선 상태 확인",
                "배기 계통 누설 및 오염 여부 점검"
            ]
        }


def render_fault_factor_panel(fault_text: str):
    insight = get_fault_insight_bundle(fault_text)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">추정 결함 요인</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size:15px; color:#c9d4df; line-height:1.75; margin-bottom:14px;">
            <span style="color:#eef3f8; font-weight:700;">진단 요약</span><br>
            {insight['summary']}
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([1.2, 1.0], gap="medium")
    with left:
        major_items = ''.join([f'<li style="margin-bottom:10px;">{x}</li>' for x in insight['major_factors']])
        st.markdown(
            f"""
            <div style="background:#141d29;border:1px solid #263241;border-radius:14px;padding:16px 18px;height:100%;">
                <div style="font-size:16px;font-weight:800;color:#eef3f8;margin-bottom:12px;">주요 요인</div>
                <ul style="margin:0;padding-left:18px;color:#c4d0dc;line-height:1.75;">{major_items}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        check_items = ''.join([f'<li style="margin-bottom:10px;">{x}</li>' for x in insight['checklist']])
        st.markdown(
            f"""
            <div style="background:#141d29;border:1px solid #263241;border-radius:14px;padding:16px 18px;height:100%;">
                <div style="font-size:16px;font-weight:800;color:#eef3f8;margin-bottom:12px;">부품 체크리스트</div>
                <ul style="margin:0;padding-left:18px;color:#c4d0dc;line-height:1.75;">{check_items}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
def train_rf_model(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy().fillna(df[feature_cols].median(numeric_only=True))
    y = df["Fault_3"].astype(int)

    model = RandomForestClassifier(
        n_estimators=450,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X)

    fi = pd.DataFrame({
        "변수": feature_cols,
        "중요도": model.feature_importances_
    }).sort_values("중요도", ascending=False).reset_index(drop=True)

    return model, pred, proba, fi


def compute_risk_score_and_level(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    df = df.copy()

    normal_mask = df[pred_col] == 0
    defect_mask = df[pred_col] != 0

    normal_md = df.loc[normal_mask, "Health_Distance"].dropna()
    max_md = df["Health_Distance"].max()

    if len(normal_md) == 0 or pd.isna(max_md):
        df["Risk_Score"] = np.nan
        df["Engine_Risk_Level"] = 0
        return df

    normal_mean = float(normal_md.mean())
    denom = max(float(max_md - normal_mean), EPS)

    risk_score = 100 * (df["Health_Distance"] - normal_mean) / denom
    risk_score = np.clip(risk_score, 0, 100)

    df["Risk_Score"] = risk_score
    df["Engine_Risk_Level"] = 0

    defect_scores = df.loc[defect_mask, "Risk_Score"].dropna().values

    if len(defect_scores) >= 20:
        q70, q90, q97 = np.quantile(defect_scores, [0.70, 0.90, 0.97])

        def level_map(x):
            if x < q70:
                return 1
            elif x < q90:
                return 2
            elif x < q97:
                return 3
            else:
                return 4
    else:
        def level_map(x):
            if x < 70:
                return 1
            elif x < 90:
                return 2
            elif x < 97:
                return 3
            else:
                return 4

    df.loc[defect_mask, "Engine_Risk_Level"] = df.loc[defect_mask, "Risk_Score"].apply(level_map).astype(int)
    return df


# =========================================================
# 시각화 함수
# =========================================================
def make_vehicle_info_panel():
    image_html = f"""
    <div style="display:flex; justify-content:center; align-items:flex-start; padding-top:12px;">
        <img src="data:image/jpeg;base64,{CHEVROLET_SAIL_BASE64}"
             style="width:70%; height:auto; border-radius:18px; object-fit:contain;" />
    </div>
    """

    with st.expander("TEST VEHICLE MODEL", expanded=True):
        c1, c2 = st.columns([1.0, 1.28], gap="large")

        with c1:
            st.markdown(
                """
                <div style="font-size:56px; font-weight:900; color:#eef3f8; margin-top:4px; line-height:1.05;">
                    Chevrolet Sail
                </div>
                <div style="font-size:20px; color:#9fb0c2; margin-top:22px; line-height:1.95;">
                    Compact Sedan / 5-Seater<br>
                    Engine: 1.4L Gasoline<br>
                    Max Power: 76 kW<br>
                    Max Torque: 131 Nm<br>
                    Transmission: 5-Speed Manual<br>
                    Dimensions: 4249 × 1690 × 1503 mm
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(image_html, unsafe_allow_html=True)


def make_angular_gauge(value, min_value, max_value, title, bar_color):
    value = 0 if pd.isna(value) else float(value)
    min_value = float(min_value)
    max_value = max(float(max_value), min_value + 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 32, "color": "#eef3f8"}},
        title={"text": title, "font": {"size": 16, "color": "#aebccb"}},
        gauge={
            "axis": {"range": [min_value, max_value], "tickcolor": "#657385"},
            "bar": {"color": bar_color},
            "bgcolor": "#1e2936",
            "borderwidth": 0,
            "steps": [
                {"range": [min_value, min_value + (max_value - min_value) * 0.5], "color": "#223141"},
                {"range": [min_value + (max_value - min_value) * 0.5, min_value + (max_value - min_value) * 0.8], "color": "#263748"},
                {"range": [min_value + (max_value - min_value) * 0.8, max_value], "color": "#2b3e52"},
            ]
        }
    ))
    fig.update_layout(
        height=245,
        margin=dict(l=12, r=12, t=38, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf3")
    )
    return fig


def make_horizontal_bar(value, min_v, max_v, title="차량 연비"):
    value = 0 if pd.isna(value) else float(value)
    min_v = 0 if pd.isna(min_v) else float(min_v)
    max_v = max(float(max_v), min_v + 1, value)

    tick_vals = np.linspace(min_v, max_v, 6)
    tick_text = [f"{v:.1f}" for v in tick_vals]

    fig = go.Figure(go.Bar(
        x=[value],
        y=[title],
        orientation="h",
        marker=dict(color="#6f9df5"),
        text=[f"{value:.2f}"],
        textposition="outside",
        hovertemplate=f"{title}: %{{x:.2f}}<extra></extra>"
    ))
    fig.update_layout(
        height=120,
        margin=dict(l=20, r=28, t=10, b=34),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf3"),
        xaxis=dict(
            range=[min_v, max_v],
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            showgrid=True,
            gridcolor="#2a3646",
            zeroline=False,
            tickfont=dict(size=11, color="#aebccb"),
            title=None
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None),
        bargap=0.4,
        showlegend=False
    )
    return fig


def make_standard_ref_bar(current_value, target_value, title, bar_color="#4fb4d8"):
    current_value = 0 if pd.isna(current_value) else float(current_value)
    target_value = 0 if pd.isna(target_value) else float(target_value)
    axis_max = max(current_value, target_value) * 1.35
    axis_max = max(axis_max, target_value + EPS, 1.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[current_value],
        y=[title],
        orientation="h",
        marker=dict(color=bar_color),
        text=[f"{current_value:.2f}"],
        textposition="outside",
        hovertemplate=f"{title}: %{{x:.2f}}<extra></extra>"
    ))
    fig.add_vline(x=target_value, line_width=2, line_dash="dash", line_color="#ff7b72")
    fig.add_annotation(
        x=target_value,
        y=0,
        text=f"기준 {target_value:.2f}",
        showarrow=False,
        yshift=18,
        font=dict(size=11, color="#ffb4ac")
    )
    fig.update_layout(
        height=110,
        margin=dict(l=10, r=28, t=8, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf3"),
        xaxis=dict(
            range=[0, axis_max],
            showgrid=True,
            gridcolor="#2a3646",
            zeroline=False,
            tickfont=dict(size=10, color="#aebccb"),
            title=None
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=12, color="#e8edf3"),
            title=None
        ),
        showlegend=False,
        bargap=0.45
    )
    return fig


def get_speed_matched_normal_df(current_row: pd.Series, normal_df: pd.DataFrame):
    if normal_df.empty or "Speed" not in normal_df.columns:
        return normal_df

    cur_speed = pd.to_numeric(current_row.get("Speed", np.nan), errors="coerce")
    if pd.isna(cur_speed):
        return normal_df

    speed_series = pd.to_numeric(normal_df["Speed"], errors="coerce")
    speed_range = float(speed_series.max() - speed_series.min()) if not speed_series.dropna().empty else 0.0
    tol = max(5.0, speed_range * 0.08)

    matched = normal_df[speed_series.between(cur_speed - tol, cur_speed + tol)].copy()
    if len(matched) < 5:
        nearest_idx = (speed_series - cur_speed).abs().sort_values().index[:10]
        matched = normal_df.loc[nearest_idx].copy()
    return matched


def make_radar_compare_chart(current_row: pd.Series, normal_df: pd.DataFrame, cols: list[str], full_df: pd.DataFrame):
    radar_priority = [
        "MAP", "TPS", "Force", "Power", "RPM", "Speed",
        "Consumption L/H", "Consumption L/100KM", "CO", "HC", "CO2", "O2", "Lambda", "AFR"
    ]
    radar_cols = [c for c in radar_priority if c in cols and c in full_df.columns]
    if not radar_cols:
        radar_cols = [c for c in cols if c in full_df.columns][:10]
    if not radar_cols:
        return go.Figure()

    current_vals, normal_vals, labels = [], [], []

    for c in radar_cols:
        series = pd.to_numeric(full_df[c], errors="coerce")
        cur = pd.to_numeric(current_row.get(c, np.nan), errors="coerce")
        norm_mean = pd.to_numeric(normal_df[c], errors="coerce").mean() if c in normal_df.columns and not normal_df.empty else np.nan

        s_min = float(series.min()) if not series.dropna().empty else 0.0
        s_max = float(series.max()) if not series.dropna().empty else 1.0
        if pd.isna(s_min):
            s_min = 0.0
        if pd.isna(s_max):
            s_max = 1.0
        if abs(s_max - s_min) < EPS:
            s_max = s_min + 1.0

        cur_norm = (float(cur) - s_min) / (s_max - s_min) if not pd.isna(cur) else 0.0
        norm_norm = (float(norm_mean) - s_min) / (s_max - s_min) if not pd.isna(norm_mean) else 0.0

        current_vals.append(float(np.clip(cur_norm, 0, 1)))
        normal_vals.append(float(np.clip(norm_norm, 0, 1)))
        labels.append(c)

    current_vals.append(current_vals[0])
    normal_vals.append(normal_vals[0])
    labels.append(labels[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normal_vals,
        theta=labels,
        fill='toself',
        name='유사 Speed 정상 평균',
        line=dict(color='#5b8def', width=2),
        fillcolor='rgba(91,141,239,0.18)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=current_vals,
        theta=labels,
        fill='toself',
        name='현재 데이터',
        line=dict(color='#e2894f', width=2),
        fillcolor='rgba(226,137,79,0.20)'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9, color='#8fa1b5'), gridcolor='#2a3646'),
            angularaxis=dict(tickfont=dict(size=9, color='#e8edf3'), gridcolor='#2a3646')
        ),
        height=520,
        margin=dict(l=10, r=10, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='center', x=0.5, font=dict(size=10, color='#dbe5f0')),
        font=dict(color='#e8edf3')
    )
    return fig

def render_info_card(title, value, subtitle="", accent="#2a3646"):
    st.markdown(
        f"""
        <div class="kpi-box" style="border-left:4px solid {accent};">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_status_card(status_text, risk_level, fault_text, prob):

    status_badge = '<span class="badge badge-green">NORMAL</span>' if status_text == "정상" else '<span class="badge badge-red">FAULT</span>'

    level_color = {
        0:"#2fbf71",
        1:"#5b8def",
        2:"#d6a74f",
        3:"#e2894f",
        4:"#e05a5a"
    }.get(risk_level, "#5b8def")

    fault_label = "정상 상태" if status_text == "정상" else fault_text

    # ===== 점검 가이드 분기 =====
    if status_text == "정상":
        guidance = "이상 징후 없음"
        accent_1 = "#2fbf71"
        accent_2 = "#5b8def"
        accent_3 = "#e05a5a"

    elif fault_text == "Rich mixture":
        guidance = "연료 분사 및 연료 공급 계통 점검"
        accent_1 = "#e05a5a"
        accent_2 = "#e2894f"
        accent_3 = "#e05a5a"

    elif fault_text in ["Lean mixture or Low voltage"]:
        guidance = "흡기 계통 및 전원 공급 상태 점검"
        accent_1 = "#e05a5a"
        accent_2 = "#d6a74f"
        accent_3 = "#e05a5a"

    else:
        guidance = "엔진 주요 센서 및 연소 상태 점검"
        accent_1 = "#5b8def"
        accent_2 = "#5b8def"
        accent_3 = "#e05a5a"

    card_shell = "position:relative; height:190px; overflow:hidden; padding-left:22px;"
    accent_style = "position:absolute; left:0; top:14px; bottom:14px; width:8px; border-radius:10px; box-shadow:0 0 12px rgba(255,255,255,0.06);"

    c1, c2, c3 = st.columns([1.0, 1.35, 1.0], gap="medium")

    with c1:
        st.markdown(
            f"""
            <div class="card" style="{card_shell}">
                <div style="{accent_style} background:linear-gradient(180deg, {accent_1} 0%, rgba(255,255,255,0.18) 100%);"></div>
                <div class="card-title">Current Diagnostic Result</div>
                <div style="font-size:34px; font-weight:800; line-height:1.12;">{status_text}</div>
                <div style="margin-top:12px;">{status_badge}</div>
                <div class="divider-line"></div>
                <div style="font-size:13px; color:#aab7c5;">현재 시점 기준 진단 결과</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="card" style="{card_shell}">
                <div style="{accent_style} background:linear-gradient(180deg, {accent_2} 0%, rgba(255,255,255,0.16) 100%);"></div>
                <div class="card-title">Estimated Fault Type</div>
                <div style="font-size:26px; font-weight:800; line-height:1.25; color:#eef3f8;">{fault_label}</div>
                <div style="margin-top:10px; font-size:14px; color:#b8c4d1; line-height:1.65;">{guidance}</div>
                <div class="divider-line"></div>
                <div style="font-size:13px; color:#aab7c5;">추정 결함 유형 기반 1차 점검 가이드</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="card" style="{card_shell}">
                <div style="{accent_style} background:linear-gradient(180deg, {accent_3} 0%, rgba(255,255,255,0.16) 100%);"></div>
                <div class="card-title">Risk & Confidence</div>
                <div style="font-size:14px; color:#9aa8b8; margin-bottom:6px;">Engine Risk Level</div>
                <div style="font-size:34px; font-weight:800; color:{level_color}; line-height:1.05;">Level {risk_level}</div>
                <div class="divider-line"></div>
                <div style="font-size:13px; color:#9aa8b8;">예측 확률</div>
                <div style="font-size:22px; font-weight:800; color:#eef3f8; margin-top:4px;">{prob:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_table_panel(title, df_view, show_cols):
    st.markdown(f'<div class="card" style="height:100%;"><div class="panel-title">{title}</div>', unsafe_allow_html=True)

    if df_view.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        view = df_view[show_cols].copy().reset_index(drop=True)
        st.dataframe(view, use_container_width=True, height=360)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 사이드바
# =========================================================
st.sidebar.markdown("### 데이터 로드")
uploaded_file = st.sidebar.file_uploader("Engine Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.sidebar.info("CSV 파일을 업로드하면 대시보드가 표시됩니다.")
    st.stop()

raw = pd.read_csv(uploaded_file)
df = add_derived_features(raw)

if "Fault_3" not in df.columns:
    st.error("업로드한 파일에는 Fault 컬럼이 필요합니다.")
    st.stop()

feature_cols = get_feature_columns(df)

st.sidebar.markdown("---")
st.sidebar.markdown("### 시간 인덱스")
time_search_idx = st.sidebar.number_input(
    "행 번호로 바로 이동",
    min_value=0,
    max_value=max(len(df) - 1, 0),
    value=0,
    step=1
)
time_idx = st.sidebar.slider(
    "각 행을 시간처럼 탐색",
    min_value=0,
    max_value=len(df) - 1,
    value=int(time_search_idx),
    step=1
)


st.sidebar.markdown("---")
st.sidebar.markdown("### 모델")
st.sidebar.caption("RandomForest 기반 정상,결함(결함 유형 별)분류")


# =========================================================
# 모델 학습 및 예측
# =========================================================
model, pred, proba, fi = train_rf_model(df, feature_cols)
df["Pred_Fault_3"] = pred.astype(int)
df["Pred_Prob"] = proba.max(axis=1)

normal_mask = df["Pred_Fault_3"] == 0
df["Health_Distance"] = compute_health_distance(df, normal_mask)

sensor_cols = feature_cols.copy()
df = compute_risk_score_and_level(df, "Pred_Fault_3")

row = df.iloc[time_idx]
pred_fault = int(row["Pred_Fault_3"])
current_status = status_label(pred_fault)
risk_level = int(row["Engine_Risk_Level"])
fault_text = fault_type_label(pred_fault)

sensor_table = make_sensor_current_table(row, sensor_cols)

exhaust_cols = [c for c in ["CO", "HC", "CO2", "O2", "Lambda", "AFR", "AFR_Deviation"] if c in sensor_table["변수"].tolist()]
performance_cols = [c for c in sensor_table["변수"].tolist() if c not in exhaust_cols]

exhaust_table = sensor_table[sensor_table["변수"].isin(exhaust_cols)].copy()
performance_table = sensor_table[sensor_table["변수"].isin(performance_cols)].copy()

rpm_val = pd.to_numeric(row.get("RPM", np.nan), errors="coerce")
speed_val = pd.to_numeric(row.get("Speed", np.nan), errors="coerce")

if "Consumption L/100KM" in row.index:
    fuel_val = pd.to_numeric(row.get("Consumption L/100KM", np.nan), errors="coerce")
    fuel_title = "차량 연비 (L/100KM)"
    fuel_col = "Consumption L/100KM"
elif "Consumption L/H" in row.index:
    fuel_val = pd.to_numeric(row.get("Consumption L/H", np.nan), errors="coerce")
    fuel_title = "차량 연비 (L/H)"
    fuel_col = "Consumption L/H"
else:
    fuel_val = 0.0
    fuel_title = "차량 연비"
    fuel_col = None

rpm_max = max(1000, float(np.nanmax(pd.to_numeric(df.get("RPM", pd.Series([1000])), errors="coerce").fillna(0))))
speed_max = max(60, float(np.nanmax(pd.to_numeric(df.get("Speed", pd.Series([60])), errors="coerce").fillna(0))))
fuel_series = pd.to_numeric(df[fuel_col], errors="coerce") if fuel_col and fuel_col in df.columns else pd.Series([0, 1])
fuel_min, fuel_max = float(fuel_series.min()), float(fuel_series.max())


# =========================================================
# 메인 화면
# =========================================================
st.markdown("## Engine QC Monitoring Dashboard")

render_status_card(
    status_text=current_status,
    risk_level=risk_level,
    fault_text=fault_text,
    prob=float(row["Pred_Prob"])
)

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
make_vehicle_info_panel()
st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

main_left, main_right = st.columns([1.05, 0.95], gap="large")

normal_reference_df = df[df["Pred_Fault_3"] == 0].copy()
if normal_reference_df.empty:
    normal_reference_df = df.copy()

speed_matched_normal_df = get_speed_matched_normal_df(row, normal_reference_df)

with main_left:
    gauge_col1, gauge_col2 = st.columns(2, gap="large")

    with gauge_col1:
        st.markdown(
            '<div class="card"><div class="panel-title">RPM</div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(
            make_angular_gauge(rpm_val, 0, rpm_max, "", "#5b8def"),
            use_container_width=True,
            config={"displayModeBar": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with gauge_col2:
        st.markdown(
            '<div class="card"><div class="panel-title">SPEED</div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(
            make_angular_gauge(speed_val, 0, speed_max, "", "#d6a74f"),
            use_container_width=True,
            config={"displayModeBar": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><div class="panel-title">차량 연비 지표(L/100KM)</div>',
        unsafe_allow_html=True
    )
    st.plotly_chart(
        make_horizontal_bar(fuel_val, fuel_min, fuel_max, fuel_title),
        use_container_width=True,
        config={"displayModeBar": False}
    )
    st.markdown('</div>', unsafe_allow_html=True)

with main_right:
    st.markdown('<div class="card"><div class="panel-title">유사 Speed 정상 데이터 VS 현재 데이터 레이더 차트</div>', unsafe_allow_html=True)
    st.plotly_chart(
        make_radar_compare_chart(row, speed_matched_normal_df, sensor_cols, df),
        use_container_width=True,
        config={"displayModeBar": False}
    )
    st.markdown(
        f'<div class="small-note" style="margin-top:4px;">비교 기준: 현재 Speed {float(speed_val) if not pd.isna(speed_val) else 0:.1f}와 유사한 정상 데이터 {len(speed_matched_normal_df)}개 평균</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

p1, p2, p3 = st.columns([1.0, 1.0, 1.15], gap="medium")
with p1:
    render_table_panel("엔진 성능 지표", performance_table, ["변수", "현재값"])
with p2:
    render_table_panel("배기가스 데이터", exhaust_table, ["변수", "현재값"])
with p3:
    st.markdown('<div class="card"><div class="panel-title">표준 배기가스 지표</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note" style="margin-bottom:10px;">현재 운행차 가솔린 검사 기준 참고값</div>', unsafe_allow_html=True)
    lambda_val = pd.to_numeric(row.get("Lambda", np.nan), errors="coerce") if "Lambda" in row.index else np.nan
    afr_val = pd.to_numeric(row.get("AFR", np.nan), errors="coerce") if "AFR" in row.index else np.nan
    co_val = pd.to_numeric(row.get("CO", np.nan), errors="coerce") if "CO" in row.index else np.nan
    hc_val = pd.to_numeric(row.get("HC", np.nan), errors="coerce") if "HC" in row.index else np.nan
    st.plotly_chart(make_standard_ref_bar(lambda_val, 1.0, "Lambda", "#4fb4d8"), use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(make_standard_ref_bar(afr_val, 14.7, "AFR", "#d6a74f"), use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(make_standard_ref_bar(co_val, 1.0, "CO", "#f08a5d"), use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(make_standard_ref_bar(hc_val, 120.0, "HC", "#86aef6"), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
render_fault_factor_panel(fault_text)
st.markdown("---")
st.caption(
    "RandomForest가 먼저 정상 / Rich mixture / Lean mixture or Low voltage를 예측하고, "
    "예측된 정상 데이터만을 기준으로 18개 컬럼 기반 Mahalanobis Health Distance를 계산한 뒤 "
    "이를 0~100 Risk Score로 스케일링하여 Engine Risk Level을 산출합니다."
)