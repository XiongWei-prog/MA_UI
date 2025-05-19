# ---------------------------  app.py  ---------------------------
# 依赖：
#   shiny ≥1.4, shinywidgets ≥0.2, plotly ≥5.8, shap ≥0.44,
#   pandas, numpy, joblib,
#   langchain ≥0.0.260, langchain-community, qianfan, python-dotenv
# ----------------------------------------------------------------
from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import plotly.express as px, plotly.graph_objects as go
import pandas as pd, numpy as np, shap, joblib, pathlib, os
# LangChain & Qianfan
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import QianfanLLMEndpoint

# ═════ 1 资产加载 & 模型初始化 ════════════════════════════════════
# 加载环境变量
load_dotenv(find_dotenv())
# 初始化 Qianfan LLM
qianfan_llm = QianfanLLMEndpoint(
    model_name="ernie-bot",
    temperature=0.4,
    max_tokens=800,
    api_key=os.getenv("BAIDU_API_KEY"),
    secret_key=os.getenv("BAIDU_SECRET_KEY")
)
# AI 分析 Prompt 模板
analysis_template = PromptTemplate.from_template(
    """
你是一名并购分析专家，需评估 {company}（股票代码：{stock_code}）在 {year} 年的并购可能性，并给出专业投资建议。

已知关键信息：
- 并购预测概率：{probability:.2%}
- 局部 SHAP 贡献前 5 特征：{top_features}
- 全局特征重要度（Top-5）：{global_imp}

请基于上述数据，生成简明的洞察和下一步建议。"""
)
chain = LLMChain(llm=qianfan_llm, prompt=analysis_template)

# 加载数据和模型
root = pathlib.Path(__file__).parent
model  = joblib.load(root / "rf.pkl")
scaler = joblib.load(root / "scaler.pkl")
# 读取并准备数据
df_full = pd.read_csv(root / "dataset.csv").drop(columns=["Unnamed: 0"], errors="ignore")
# 特征列对齐
explainer = shap.TreeExplainer(model)
candidate_cols = (
    list(getattr(explainer, "feature_names", []) or [])
    or list(getattr(model, "feature_names_in_", []) or [])
)
if not candidate_cols:
    candidate_cols = list(df_full.columns.drop(["MA","Stkcd","year"]))
FEATURE_COLS = candidate_cols
# 补齐可能缺失的特征列
for c in FEATURE_COLS:
    if c not in df_full.columns:
        df_full[c] = 0.0
# 年份列表
years_all = sorted(df_full["year"].unique(), reverse=True)

# ─── 帮助函数：鲁棒提取 SHAP 向量 ─────────────────────────
def extract_shap_vector(shap_raw, n_feat):
    arr = np.array(shap_raw)
    arr = arr.squeeze()
    if arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] == n_feat:
            return arr[1]
        if arr.shape[1] == 2 and arr.shape[0] == n_feat:
            return arr[:, 1]
        return arr.flatten()[:n_feat]
    if arr.ndim == 1:
        return arr[:n_feat]
    return arr.flatten()[:n_feat]

# ═════ 2 UI 布局 ════════════════════════════════════════════════
app_ui = ui.page_sidebar(
    # —— 左侧控制区 ——  
    ui.sidebar(
        ui.input_text("code", "股票代码", placeholder="600519"),
        ui.input_select(
            "year", "年份",
            {str(y): str(y) for y in years_all},
            selected=str(years_all[0])
        ),
        ui.output_text_verbatim("warn"),
    ),

    # —— 第一行 ——  
    ui.row(
        # 左侧 6 栅格：Watchlist  
        ui.column(
            6,
            ui.card(
                ui.card_header("Watchlist — Top10 并购概率"),
                ui.output_table("watch_tbl")
            )
        ),
        # 右侧 6 栅格：全局特征重要度  
        ui.column(
            6,
            ui.card(
                ui.card_header("全局特征重要度 (Top-20)"),
                output_widget("global_imp_plot")
            )
        )
    ),

    # —— 第二行 ——  
    ui.row(
        # 左侧 6 栅格：单股预测 & SHAP  
        ui.column(
            6,
            ui.card(
                ui.card_header("并购概率 & SHAP 解释"),
                ui.output_text("pred_text"),
                output_widget("local_shap")
            )
        ),
        # 右侧 6 栅格：AI 深度解读  
        ui.column(
            6,
            ui.card(
                ui.card_header("🧠 AI 深度解读"),
                ui.output_text("llm_analysis")
            )
        )
    )
)

# ═════ 3 Server 逻辑 ═════════════════════════════════════════════
def server(input, output, session):
    @reactive.calc
    def df_year():
        return df_full[df_full["year"] == int(input.year())].copy()

    @reactive.calc
    def watch_df():
        df = df_year().copy()
        df["Prob"] = model.predict_proba(scaler.transform(df[FEATURE_COLS]))[:, 1]
        last3 = df_full[df_full["year"].between(int(input.year())-2, int(input.year()))]
        df["MA_3yr"] = (last3.groupby("Stkcd")["MA"].sum()
                          .reindex(df["Stkcd"]).fillna(0).astype(int).values)
        return (df[["Stkcd","Prob","MA_3yr"]]
                .sort_values("Prob", ascending=False)
                .head(10)
                .reset_index(drop=True))

    @output
    @render.table
    def watch_tbl():
        df = watch_df().copy()
        # 增加一个百分比列
        df["Prob%"] = (df["Prob"] * 100).round(2).astype(str) + " %"
        # 只显示三列
        return df[["Stkcd", "Prob%", "MA_3yr"]]

    @reactive.calc
    def current_pred():
        df0 = df_year()
        row = df0[df0["Stkcd"].astype(str) == input.code().strip()]
        if row.empty:
            return None
        # 预测概率
        x = scaler.transform(row[FEATURE_COLS])
        prob = model.predict_proba(x)[0, 1]
        # SHAP 向量
        shap_raw = explainer.shap_values(x)
        n_feat   = len(FEATURE_COLS)
        shap_vec = extract_shap_vector(shap_raw, n_feat)
        # 局部 top5
        contrib5 = (pd.Series(shap_vec, index=FEATURE_COLS)
                       .abs().sort_values(ascending=False).head(5))
        top5     = "; ".join(f"{f}={v:.4f}" for f,v in contrib5.items())
        # 全局 top5
        glob5s   = (pd.Series(model.feature_importances_, index=FEATURE_COLS)
                       .sort_values(ascending=False).head(5))
        glob_str = "; ".join(f"{f}={v:.4f}" for f,v in glob5s.items())
        return int(input.year()), input.code().strip(), prob, shap_vec, top5, glob_str

    @output
    @render.text
    def warn():
        if not input.code().strip():
            return "请输入股票代码"
        if current_pred() is None:
            return "⚠ 该股票在所选年份无数据"
        return ""

    @output
    @render.text
    def pred_text():
        res = current_pred()
        if not res:
            return ""
        year, code, prob, *_ = res
        tag = "⚠ 高并购概率" if prob >= 0.5 else "√ 并购概率低"
        return f"{year} 年预测并购概率：{prob:.2%} → {tag}"

    @output
    @render_widget
    def local_shap():
        res = current_pred()
        if not res:
            return None
        _, _, _, shap_vec, _, _ = res
        # 取前15并排序
        contrib = (pd.Series(shap_vec, index=FEATURE_COLS)
                     .abs().sort_values(ascending=False).head(15))
        fig = px.bar(
            x=contrib.values * np.sign(shap_vec[:len(contrib)]),
            y=contrib.index,
            orientation="h",
            color=contrib.values * np.sign(shap_vec[:len(contrib)]),
            color_continuous_scale="RdBu",
            labels={"x":"对 logit 的贡献","y":"特征"},
            title=f"SHAP Top-{len(contrib)}"
        )
        fig.update_layout(coloraxis_showscale=False,
                          height=430,
                          yaxis_categoryorder="total ascending")
        return fig

    @output
    @render_widget
    def global_imp_plot():
        global_imp = (pd.Series(model.feature_importances_, index=FEATURE_COLS)
                        .sort_values(ascending=False)
                        .reset_index()
                        .rename(columns={"index":"Feature", 0:"Importance"}))
        fig = px.bar(
            global_imp.head(20), x="Importance", y="Feature",
            orientation="h",
            title="随机森林全局特征重要度 (Top-20)"
        )
        fig.update_layout(height=430,
                          yaxis_categoryorder="total ascending")
        return fig

    @output
    @render.text
    def llm_analysis():
        res = current_pred()
        if not res:
            return ""
        year, code, prob, _, top5, glob5 = res
        # 构造 Prompt 并调用大模型
        reply = chain.run({
            "company": code,
            "stock_code": code,
            "year": year,
            "probability": prob,
            "top_features": top5,
            "global_imp": glob5
        })
        return reply

# ═════ 4 实例化 & 运行 ════════════════════════════════════════════
app = App(app_ui, server)



