"""
JNJ LLM Tool — Flask Web Application
Dual model: Gemini 2.5 Flash + Claude Sonnet
Feedback loop: user feedback → single LLM call → new dashboard
Run:  python3 app.py
Open: http://127.0.0.1:5000
"""

from dotenv import load_dotenv
load_dotenv()

from google import genai as google_genai
import anthropic
from flask import Flask, request, jsonify, render_template
import os, re, json, warnings, datetime, uuid, tempfile
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────

gemini_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# FLASK SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "jnj_llm_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

_df_store: Dict[str, Dict[str, pd.DataFrame]] = {}
_history_store: Dict[str, List[dict]] = {}
_profile_store: Dict[str, dict] = {}
_code_store: Dict[str, str] = {}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "nil", "-", "--", "?", "unknown"}

CLEANING_POLICY = {
    "allow_trim_whitespace":            True,
    "allow_standardize_missing_tokens": True,
    "allow_deduplicate_rows":           True,
    "allow_type_cast_numeric":          True,
    "allow_type_cast_datetime":         True,
    "allow_drop_cols":                  False,
    "allow_drop_rows":                  False,
    "allow_impute":                     False,
    "allow_outlier_capping":            False,
    "max_drop_row_pct":                 0.005,
    "max_drop_col_pct":                 0.20,
    "max_impute_missing_pct":           0.05,
    "min_parse_success_numeric":        0.90,
    "min_parse_success_datetime":       0.80,
}

GOAL_MAP = {
    "financial":   "financial dashboard and executive reporting",
    "forecasting": "time-series feature engineering and forecasting preparation",
    "anomaly":     "anomaly and outlier detection in financial data",
    "general":     "general exploratory data analysis",
}

# ─────────────────────────────────────────────────────────────────────────────
# PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def _infer_type_group(series: pd.Series, policy: dict) -> str:
    if pd.api.types.is_numeric_dtype(series):        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series): return "datetime"
    if pd.api.types.is_bool_dtype(series):           return "boolean"
    if policy.get("allow_type_cast_numeric"):
        coerced = pd.to_numeric(series.dropna(), errors="coerce")
        if len(series.dropna()) > 0 and coerced.notna().sum() / len(series.dropna()) >= policy["min_parse_success_numeric"]:
            return "numeric-candidate"
    if policy.get("allow_type_cast_datetime"):
        try:
            coerced_dt = pd.to_datetime(series.dropna(), errors="coerce", infer_datetime_format=True)
            if len(series.dropna()) > 0 and coerced_dt.notna().sum() / len(series.dropna()) >= policy["min_parse_success_datetime"]:
                return "datetime-candidate"
        except Exception:
            pass
    return "categorical"

def _profile_numeric(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"missing_pct": 1.0, "note": "all_missing"}
    q = np.percentile(s, [1, 5, 25, 50, 75, 95, 99])
    iqr = float(q[4] - q[2])
    mad = float(np.median(np.abs(s - float(q[3]))))
    return {
        "missing_count":    int(series.isna().sum()),
        "missing_pct":      round(float(series.isna().mean()), 4),
        "mean":             round(float(s.mean()), 4),
        "median":           round(float(s.median()), 4),
        "std":              round(float(s.std()), 4),
        "min":              round(float(s.min()), 4),
        "max":              round(float(s.max()), 4),
        "q25": round(float(q[2]),4), "q50": round(float(q[3]),4),
        "q75": round(float(q[4]),4), "q95": round(float(q[5]),4),
        "iqr":              round(iqr, 4),
        "skewness":         round(float(s.skew()), 4),
        "kurtosis":         round(float(s.kurtosis()), 4),
        "zero_pct":         round(float((s == 0).mean()), 4),
        "negative_pct":     round(float((s < 0).mean()), 4),
        "outliers_iqr_1_5": int(((s < q[2]-1.5*iqr) | (s > q[4]+1.5*iqr)).sum()),
        "outliers_iqr_3_0": int(((s < q[2]-3.0*iqr) | (s > q[4]+3.0*iqr)).sum()),
        "outliers_mad":     int(((np.abs(s-float(q[3]))/(mad*1.4826+1e-9))>3.5).sum()) if mad>0 else 0,
    }

def _profile_categorical(series: pd.Series, n_rows: int) -> dict:
    s = series.dropna()
    n_unique = int(s.nunique())
    vc = s.value_counts()
    return {
        "missing_pct":       round(float(series.isna().mean()), 4),
        "n_unique":          n_unique,
        "cardinality_ratio": round(n_unique / max(n_rows,1), 4),
        "top_10_values":     {str(k): int(v) for k, v in vc.head(5).items()},
        "rare_rate":         round(float((vc/max(len(s),1)<0.01).sum()/max(n_unique,1)), 4),
        "id_like":           (n_unique/max(n_rows,1)) > 0.9,
    }

def _profile_datetime(series: pd.Series) -> dict:
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True).dropna()
    if len(s) == 0:
        return {"missing_pct": 1.0, "note": "all_missing"}
    diffs = s.sort_values().diff().dropna()
    med_days = float(diffs.dt.total_seconds().median()/86400) if len(diffs)>0 else None
    return {
        "missing_pct": round(float(series.isna().mean()), 4),
        "min": str(s.min()), "max": str(s.max()),
        "median_diff_days": round(med_days, 2) if med_days else None,
    }

def _top_correlations(df: pd.DataFrame, k: int = 8) -> List[dict]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2: return []
    corr = df[num_cols].corr()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append({"col_a": corr.columns[i], "col_b": corr.columns[j], "pearson_r": round(float(v),4)})
    pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
    return pairs[:k]

def compute_eda_profile(df_dict: Dict[str, pd.DataFrame], policy: dict) -> dict:
    profile = {}
    for sheet_name, df in df_dict.items():
        n_rows, n_cols = df.shape
        type_groups = {col: _infer_type_group(df[col], policy) for col in df.columns}
        type_counts: Dict[str, int] = {}
        for tg in type_groups.values():
            type_counts[tg] = type_counts.get(tg, 0) + 1
        col_profiles: Dict[str, dict] = {}
        for col, tg in type_groups.items():
            if tg in ("numeric","numeric-candidate"):
                col_profiles[col] = {"type": tg, **_profile_numeric(df[col])}
            elif tg in ("datetime","datetime-candidate"):
                col_profiles[col] = {"type": tg, **_profile_datetime(df[col])}
            else:
                col_profiles[col] = {"type": tg, **_profile_categorical(df[col], n_rows)}
        miss_series = df.isna().mean().sort_values(ascending=False)
        sample_rows = df.head(3).replace({np.nan: None}).to_dict(orient="records")
        for row in sample_rows:
            for k, v in row.items():
                if isinstance(v, str) and len(v) > 80:
                    row[k] = v[:80] + "..."
        dup_rows = int(df.duplicated().sum())
        missing_total = int(df.isna().sum().sum())
        profile[sheet_name] = {
            "n_rows": n_rows, "n_cols": n_cols,
            "memory_bytes":        int(df.memory_usage(deep=True).sum()),
            "duplicate_rows":      dup_rows,
            "duplicate_row_pct":   round(dup_rows/max(n_rows,1), 4),
            "missing_cells_total": missing_total,
            "missing_cell_pct":    round(missing_total/max(n_rows*n_cols,1), 4),
            "type_counts":         type_counts,
            "top_missing_cols":    {c: round(float(v),4) for c,v in miss_series[miss_series>0].head(8).items()},
            "column_profiles":     col_profiles,
            "top_correlations":    _top_correlations(df),
            "sample_rows":         sample_rows,
        }
    return profile

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def apply_safe_cleaning(df_dict: Dict[str, pd.DataFrame], policy: dict) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    cleaning_log: List[dict] = []
    cleaned: Dict[str, pd.DataFrame] = {}
    for sheet_name, df in df_dict.items():
        df = df.copy()
        if policy.get("allow_trim_whitespace"):
            old_cols = list(df.columns)
            df.columns = [str(c).strip() for c in df.columns]
            changed = sum(a!=b for a,b in zip(old_cols, df.columns))
            if changed:
                cleaning_log.append({"sheet": sheet_name, "operation": "trim_column_names",
                    "cells_changed": changed, "rows_dropped": 0,
                    "rationale": f"Stripped whitespace from {changed} column name(s)."})
            trimmed = 0
            for col in df.select_dtypes(include=["object"]).columns:
                before = df[col].copy()
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                trimmed += int((df[col] != before).sum())
            if trimmed:
                cleaning_log.append({"sheet": sheet_name, "operation": "trim_string_cells",
                    "cells_changed": trimmed, "rows_dropped": 0,
                    "rationale": f"Trimmed whitespace from {trimmed} string cell(s)."})
        if policy.get("allow_standardize_missing_tokens"):
            count = 0
            for col in df.select_dtypes(include=["object"]).columns:
                mask = df[col].apply(lambda x: isinstance(x, str) and x.strip().lower() in MISSING_TOKENS)
                count += int(mask.sum())
                df.loc[mask, col] = np.nan
            if count:
                cleaning_log.append({"sheet": sheet_name, "operation": "standardize_missing_tokens",
                    "cells_changed": count, "rows_dropped": 0,
                    "rationale": f"Replaced {count} sentinel missing-value strings with NaN."})
        if policy.get("allow_deduplicate_rows"):
            before = len(df)
            df = df.drop_duplicates()
            dropped = before - len(df)
            if dropped:
                cleaning_log.append({"sheet": sheet_name, "operation": "deduplicate_rows",
                    "cells_changed": 0, "rows_dropped": dropped,
                    "rationale": f"Removed {dropped} exact duplicate row(s)."})
        if policy.get("allow_type_cast_numeric"):
            for col in df.select_dtypes(include=["object"]).columns:
                coerced = pd.to_numeric(df[col], errors="coerce")
                success = coerced.notna().sum() / max(df[col].notna().sum(), 1)
                if success >= policy["min_parse_success_numeric"]:
                    df[col] = coerced
                    cleaning_log.append({"sheet": sheet_name, "operation": "type_cast_numeric",
                        "column": col, "cells_changed": int(coerced.notna().sum()), "rows_dropped": 0,
                        "rationale": f"Cast '{col}' to numeric ({success:.0%} parse success)."})
        if policy.get("allow_type_cast_datetime"):
            for col in df.select_dtypes(include=["object"]).columns:
                try:
                    coerced = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    success = coerced.notna().sum() / max(df[col].notna().sum(), 1)
                    if success >= policy["min_parse_success_datetime"]:
                        df[col] = coerced
                        cleaning_log.append({"sheet": sheet_name, "operation": "type_cast_datetime",
                            "column": col, "cells_changed": int(coerced.notna().sum()), "rows_dropped": 0,
                            "rationale": f"Cast '{col}' to datetime ({success:.0%} parse success)."})
                except Exception:
                    pass
        cleaned[sheet_name] = df
    return cleaned, cleaning_log

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_SCHEMA = """{
  "report_title": "string",
  "executive_summary": "string with concrete numbers",
  "global_insights": ["string"],
  "per_sheet": {
    "<sheet_name>": {
      "summary": "string",
      "data_quality_issues": [{"type":"string","severity":"high|medium|low","evidence":"string","suggested_action":"string","requires_user_approval":true}],
      "recommended_cleaning": [{"operation":"string","columns":["string"],"why":"string","requires_user_approval":false}]
    }
  },
  "python_code": "string"
}"""

REFINE_SCHEMA = """{
  "summary": "one sentence describing what changed",
  "python_code": "string — updated dashboard code"
}"""

def build_system_prompt() -> str:
    return (
        "You are a senior quantitative analyst specializing in financial data.\n"
        "Analyze the EDA profile and return a structured JSON report.\n\n"
        "STRICT RULES:\n"
        "1. Return STRICT JSON ONLY. No markdown. No code fences.\n"
        "2. python_code must define fig using make_subplots(rows=2, cols=3) — exactly 6 charts.\n"
        "3. python_code must NOT contain: imports, file I/O, eval(), exec().\n"
        "4. Access data via df_dict['SheetName']. Use df.iloc[:,i] when column names are uncertain.\n"
        "5. Every data quality issue MUST cite numeric evidence.\n"
        "6. python_code must use template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#161b22'.\n\n"
        f"REQUIRED OUTPUT SCHEMA:\n{OUTPUT_SCHEMA}"
    )

def build_refine_system_prompt() -> str:
    return (
        "You are a senior quantitative analyst updating a Plotly dashboard based on user feedback.\n\n"
        "STRICT RULES:\n"
        "1. Return STRICT JSON ONLY. No markdown. No code fences.\n"
        "2. python_code must define fig using make_subplots(rows=2, cols=3) — exactly 6 charts.\n"
        "3. python_code must NOT contain: imports, file I/O, eval(), exec().\n"
        "4. Access data via df_dict['SheetName']. Use df.iloc[:,i] when column names are uncertain.\n"
        "5. python_code must use template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#161b22'.\n"
        "6. Apply the user's instructions directly.\n\n"
        f"REQUIRED OUTPUT SCHEMA:\n{REFINE_SCHEMA}"
    )

def build_user_prompt(profile: dict, policy: dict, user_goal: str, cleaning_log: Optional[List[dict]] = None) -> str:
    compact_profile = {}
    for sheet, data in profile.items():
        compact = dict(data)
        cp = {}
        for col, cp_data in data.get("column_profiles", {}).items():
            cp_entry = dict(cp_data)
            if "top_10_values" in cp_entry:
                cp_entry["top_10_values"] = dict(list(cp_entry["top_10_values"].items())[:3])
            cp[col] = cp_entry
        compact["column_profiles"] = cp
        compact_profile[sheet] = compact
    parts = [
        f"USER GOAL: {user_goal}",
        f"\nEDA PROFILE:\n{json.dumps(compact_profile, indent=2, default=str)}",
    ]
    if cleaning_log:
        parts.append(f"\nCLEANING LOG:\n{json.dumps(cleaning_log, indent=2)}")
    parts.append("\nGenerate the JSON report with a professional dark-themed Plotly dashboard.")
    return "\n".join(parts)

def build_refine_prompt(feedback: str, prior_code: str, history: List[dict]) -> str:
    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\nCONVERSATION HISTORY:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in recent
        ) + "\n"
    return (
        f"USER INSTRUCTIONS:\n{feedback}\n"
        f"{history_text}"
        f"\nCURRENT DASHBOARD CODE:\n{prior_code}\n\n"
        "Update the dashboard to follow the user's instructions. Return JSON only."
    )

# ─────────────────────────────────────────────────────────────────────────────
# JSON REPAIR
# ─────────────────────────────────────────────────────────────────────────────

def _repair_and_parse(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text).strip()
    if not text.startswith("{"):
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s: raise ValueError("No JSON object found.")
        text = text[s:e+1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except Exception:
        for key in ("python_code",):
            m = re.search(rf'"{key}"\s*:\s*"(.*?)(?<!\\)"\s*[,}}]', text, flags=re.DOTALL)
            if m:
                py = m.group(1).replace("\\n","\n").replace('\\"','"')
                return {"report_title":"JNJ LLM Tool","executive_summary":"Report generated.",
                        "global_insights":[],"per_sheet":{},"python_code":py,"summary":"Updated."}
        raise

# ─────────────────────────────────────────────────────────────────────────────
# LLM CALLS
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(system_prompt: str, user_prompt: str, max_retries: int = 2) -> dict:
    last_error = None
    for attempt in range(1, max_retries+1):
        try:
            prompt = system_prompt + "\n\n" + user_prompt
            if attempt > 1:
                prompt += "\n\nPREVIOUS ATTEMPT FAILED. Use df.iloc[:,i] everywhere. Return valid JSON only."
            response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            result = _repair_and_parse((response.text or "").strip())
            if not result.get("python_code","").strip():
                raise ValueError("python_code is empty.")
            return result
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Gemini failed: {last_error}")

def call_claude(system_prompt: str, user_prompt: str, max_retries: int = 2) -> dict:
    last_error = None
    for attempt in range(1, max_retries+1):
        try:
            up = user_prompt
            if attempt > 1:
                up += "\n\nPREVIOUS ATTEMPT FAILED. Use df.iloc[:,i] everywhere. Return valid JSON only."
            response = claude_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=6000,
                system=system_prompt,
                messages=[{"role": "user", "content": up}],
            )
            text = response.content[0].text if response.content else ""
            result = _repair_and_parse(text.strip())
            if not result.get("python_code","").strip():
                raise ValueError("python_code is empty.")
            return result
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Claude failed: {last_error}")

def call_llm(model: str, system_prompt: str, user_prompt: str) -> dict:
    if model == "claude":
        return call_claude(system_prompt, user_prompt)
    return call_gemini(system_prompt, user_prompt)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def run_generated_code(python_code: str, df_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    lines = [l for l in python_code.splitlines() if not re.match(r"\s*(import |from .+ import)", l)]
    code = "\n".join(lines).replace("fig.show()","")
    banned = [(r"^\s*(import |from .+ import)",re.MULTILINE),(r"\bos\b",0),(r"\bsubprocess\b",0),
              (r"\brequests\b",0),(r"\burllib\b",0),(r"\bopen\s*\(",0),
              (r"\beval\s*\(",0),(r"\bexec\s*\(",0),(r"__[a-z]+__",0)]
    for pattern, flags in banned:
        if re.search(pattern, code, flags=flags):
            raise ValueError("Unsafe pattern detected.")
    ns = {"df_dict":df_dict,"go":go,"px":px,"make_subplots":make_subplots,"np":np,"pd":pd}
    exec(code, ns)
    if "fig" not in ns: raise ValueError("Generated code did not produce 'fig'.")
    return ns["fig"]

def build_plotly_html(fig: go.Figure) -> str:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22", height=680,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig.to_html(
        full_html=False, include_plotlyjs="cdn",
        config={"displayModeBar": True, "responsive": True},
    )

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        return jsonify({"error": "Unsupported format. Use .csv, .xlsx, or .xls"}), 400
    upload_id = str(uuid.uuid4())
    save_path = UPLOAD_FOLDER / f"{upload_id}{ext}"
    f.save(str(save_path))
    try:
        if ext == ".csv":
            df_dict = {"Main": pd.read_csv(str(save_path))}
        else:
            xls = pd.ExcelFile(str(save_path))
            df_dict = {}
            for sheet in xls.sheet_names:
                try:
                    df_dict[sheet] = pd.read_excel(str(save_path), sheet_name=sheet)
                except Exception:
                    pass
            if not df_dict:
                return jsonify({"error": "Could not load any sheets."}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    _df_store[upload_id] = df_dict
    _history_store[upload_id] = []
    _profile_store[upload_id] = {}
    _code_store[upload_id] = ""
    sheets_info = [{"name": name, "rows": df.shape[0], "cols": df.shape[1]} for name, df in df_dict.items()]
    return jsonify({"upload_id": upload_id, "filename": f.filename, "sheets": sheets_info})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    upload_id       = data.get("upload_id")
    goal_key        = data.get("goal", "financial")
    apply_clean     = data.get("apply_cleaning", True)
    selected_sheets = data.get("selected_sheets", None)
    model           = data.get("model", "gemini")

    if not upload_id or upload_id not in _df_store:
        return jsonify({"error": "Session expired. Please re-upload your file."}), 400

    df_dict = _df_store[upload_id]
    if selected_sheets:
        df_dict = {k: v for k, v in df_dict.items() if k in selected_sheets}
        if not df_dict:
            return jsonify({"error": "No valid sheets selected."}), 400

    user_goal = GOAL_MAP.get(goal_key, GOAL_MAP["financial"])

    try:
        cleaning_log: List[dict] = []
        if apply_clean:
            df_dict_clean, cleaning_log = apply_safe_cleaning(df_dict, CLEANING_POLICY)
        else:
            df_dict_clean = df_dict

        profile_clean = compute_eda_profile(df_dict_clean, CLEANING_POLICY)
        _profile_store[upload_id] = profile_clean

        user_prompt = build_user_prompt(profile_clean, CLEANING_POLICY, user_goal, cleaning_log)
        result = call_llm(model, build_system_prompt(), user_prompt)

        python_code = result.get("python_code", "")
        _code_store[upload_id] = python_code
        _history_store[upload_id] = [
            {"role": "user",      "content": f"Initial analysis: {user_goal}"},
            {"role": "assistant", "content": result.get("executive_summary", "")[:300]},
        ]

        fig = run_generated_code(python_code, df_dict_clean)
        plotly_html = build_plotly_html(fig)

        total_rows = sum(p["n_rows"] for p in profile_clean.values())
        total_cols = sum(p["n_cols"] for p in profile_clean.values())
        total_dup  = sum(p["duplicate_rows"] for p in profile_clean.values())
        avg_miss   = round(float(np.mean([p["missing_cell_pct"] for p in profile_clean.values()])) * 100, 1)

        return jsonify({
            "status":            "ok",
            "iteration":         1,
            "model":             model,
            "report_title":      result.get("report_title", "JNJ LLM Tool Report"),
            "executive_summary": result.get("executive_summary", ""),
            "global_insights":   result.get("global_insights", []),
            "per_sheet":         result.get("per_sheet", {}),
            "cleaning_log":      cleaning_log,
            "plotly_html":       plotly_html,
            "metrics": {
                "sheets":       len(profile_clean),
                "rows":         total_rows,
                "cols":         total_cols,
                "duplicates":   total_dup,
                "missing_avg":  avg_miss,
                "cleaning_ops": len(cleaning_log),
            },
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/refine", methods=["POST"])
def refine():
    """Single LLM call: feedback + prior code + history → new dashboard."""
    data      = request.json or {}
    upload_id = data.get("upload_id")
    feedback  = data.get("feedback", "").strip()
    model     = data.get("model", "gemini")
    iteration = data.get("iteration", 1)

    if not upload_id or upload_id not in _df_store:
        return jsonify({"error": "Session expired. Please re-upload your file."}), 400
    if not feedback:
        return jsonify({"error": "Please provide feedback or instructions."}), 400

    prior_code = _code_store.get(upload_id, "")
    if not prior_code:
        return jsonify({"error": "No prior analysis found. Run analysis first."}), 400

    df_dict = _df_store[upload_id]
    history = _history_store.get(upload_id, [])

    try:
        user_prompt = build_refine_prompt(feedback, prior_code, history)
        result = call_llm(model, build_refine_system_prompt(), user_prompt)

        new_code = result.get("python_code", "")
        summary  = result.get("summary", "Dashboard updated.")

        fig = run_generated_code(new_code, df_dict)
        plotly_html = build_plotly_html(fig)

        _code_store[upload_id] = new_code
        history.append({"role": "user",      "content": feedback})
        history.append({"role": "assistant", "content": summary})
        _history_store[upload_id] = history[-10:]

        return jsonify({
            "status":      "ok",
            "iteration":   iteration + 1,
            "model":       model,
            "summary":     summary,
            "plotly_html": plotly_html,
            "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  JNJ LLM Tool — Web Server Starting")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 55)
    app.run(debug=True, port=5000, host='127.0.0.1')
