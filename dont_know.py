#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pharma Bankruptcy Risk Pipeline (Modular Skeleton)
-------------------------------------------------
- می‌سازد: معیار جدید مبتنی بر لاجیت L1 (قابل‌توضیح) + (اختیاری) XGBoost
- مقایسه: با Z-Score آلتمن (نسخه‌های EM/Z'/Z'') به‌صورت پیوسته و آستانه‌ای
- پایداری: Split زمانی، کالیبراسیون، Bootstrap CI، آزمون DeLong برای AUROC
- هدف: اسکلت تمیز و قابل‌تعویض ماژول‌ها؛ هر بخش را می‌توان مستقل عوض/بهینه کرد

نکته مهم: این فایل به‌عنوان «اسکلت» طراحی شده و با داده واقعی شما پر می‌شود.
TODOها را جست‌وجو کنید و مطابق داده‌تان تکمیل کنید.

نیازمندی‌ها: pandas, numpy, scikit-learn, scipy, imbalanced-learn (اختیاری), xgboost (اختیاری)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, f1_score,
    precision_recall_fscore_support, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.utils import resample

try:
    import xgboost as xgb  # optional
except Exception:
    xgb = None

from scipy import stats

# ---------------------------
# 0) Config
# ---------------------------

@dataclass
class Config:
    data_path: str = "./your_data.csv"  # TODO: مسیر فایل داده شما
    id_col: str = "firm_id"             # TODO: شناسه شرکت
    time_col: str = "fiscal_year"       # TODO: ستون زمان (سال/تاریخ)
    label_col: str = "label"            # 1=ورشکستگی در افق T، 0=عدم ورشکستگی
    features: List[str] = None           # TODO: فهرست متغیرهای ویژگی عددی
    # گزینه‌های پیش‌پردازش
    group_standardize_cols: List[str] = None  # مثل ["fiscal_year"] یا ["industry","fiscal_year"]
    winsorize_q: float = 0.01
    impute_strategy: str = "median_by_year"  # یا "median_global"
    # Split زمانی
    train_until: Any = 1398  # TODO: آخرین سال آموزش
    valid_until: Any = 1400  # TODO: آخرین سال اعتبارسنجی؛ بقیه Test
    # مدل
    model_name: str = "xgboost"  # یا "xgboost"logit_l1
    class_weight: Optional[Dict[int, float]] = None  # مثل {0:1, 1:5}
    # لاجیت L1
    C: float = 0.5
    max_iter: int = 500
    # XGBoost
    xgb_params: Dict[str, Any] = None
    # کالیبراسیون
    calibrate: bool = True
    calibration_method: str = "isotonic"  # یا "sigmoid"
    # آستانه
    threshold_strategy: str = "youden"  # "youden" | "f1" | "cost"
    cost_fp: float = 1.0
    cost_fn: float = 5.0
    # Bootstrap/DeLong
    n_boot: int = 500
    random_state: int = 42
    # Altman Z mapping
    altman_variant: str = "EM"  # "EM" | "Zprime" | "Zdoubleprime"
    altman_mapping: Dict[str, str] = None  # کلیدها پایین تعریف می‌شوند

    def __post_init__(self):
        if self.features is None:
            self.features = [
                # نمونه اولیه؛ شما جایگزین کنید
                "current_ratio", "quick_ratio", "cash_ratio",
                "debt_to_assets", "debt_to_equity", "st_debt_to_assets",
                "roa", "roe", "ebit_to_assets", "gross_margin", "operating_margin",
                "asset_turnover", "inventory_turnover", "days_inventory", "dso",
                "accruals_to_assets", "cfo_to_assets", "sales_growth",
                "ebitda_growth", "margin_cov", "import_dependency", "days_inventory_key_apis",
                "fx_exposure", "govt_sales_share", "insurance_receivables_delay",
                "log_assets"
            ]
        if self.group_standardize_cols is None:
            self.group_standardize_cols = [self.time_col]
        if self.xgb_params is None:
            self.xgb_params = {
                "n_estimators": 400,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_jobs": -1,
                "random_state": self.random_state,
            }
        if self.altman_mapping is None:
            self.altman_mapping = {
                # TODO: نام ستون‌های متناظر در داده شما
                # همه به صورت سطح سالانه و بر مبنای ترازنامه/سود و زیان و ارزش بازار
                "working_capital": "working_capital",
                "retained_earnings": "retained_earnings",
                "ebit": "ebit",
                "market_value_equity": "mve",  # ارزش بازار حقوق صاحبان سهام
                "book_equity": "equity_book",
                "sales": "sales",
                "total_assets": "total_assets",
                "total_liabilities": "total_liabilities",
            }

# ---------------------------
# 1) Utilities
# ---------------------------

def temporal_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    # اگر time_col تاریخ است، به سال تبدیل کنید (در صورت نیاز)
    if np.issubdtype(df[cfg.time_col].dtype, np.datetime64):
        df["_year"] = pd.to_datetime(df[cfg.time_col]).dt.year
        tcol = "_year"
    else:
        tcol = cfg.time_col
    train = df[df[tcol] <= cfg.train_until]
    valid = df[(df[tcol] > cfg.train_until) & (df[tcol] <= cfg.valid_until)]
    test = df[df[tcol] > cfg.valid_until]
    return train, valid, test


def winsorize_by_group(df: pd.DataFrame, cols: List[str], group_cols: List[str], q: float) -> pd.DataFrame:
    df = df.copy()
    if not cols:
        return df
    def _clip(group: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            if c in group.columns:
                lo, hi = group[c].quantile(q), group[c].quantile(1 - q)
                group[c] = group[c].clip(lo, hi)
        return group
    return df.groupby(group_cols, group_keys=False).apply(_clip)


def impute_missing(df: pd.DataFrame, cols: List[str], strategy: str, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if strategy == "median_by_year" and group_cols:
        med = df.groupby(group_cols)[cols].transform("median")
        df[cols] = df[cols].fillna(med)
    else:
        med = df[cols].median()
        df[cols] = df[cols].fillna(med)
    return df


def standardize_by_group(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                          cols: List[str], group_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    استانداردسازی داخل گروه (مثلاً سال-صنعت): z = (x - mean_group) / std_group
    روی Train آمار را می‌گیریم و همان را روی Val/Test اعمال می‌کنیم تا Leakage نداشته باشیم.
    """
    train = train.copy(); valid = valid.copy(); test = test.copy()

    # محاسبه میانگین/انحراف معیار گروهی روی Train
    grp = train.groupby(group_cols)
    means = grp[cols].transform("mean")
    stds = grp[cols].transform("std").replace(0, 1.0)
    train_z = (train[cols] - means) / stds
    for c in cols:
        train[f"z_{c}"] = train_z[c]

    # برای Val/Test از آمار گروهی Train نزدیک‌ترین گروه استفاده می‌کنیم
    # راه ساده: از آمار کلی Train استفاده کنیم؛ برای پایداری می‌توان با Merge گروهی پیاده‌سازی کرد
    global_means = train[cols].mean()
    global_stds = train[cols].std().replace(0, 1.0)

    valid_z = (valid[cols] - global_means) / global_stds
    test_z = (test[cols] - global_means) / global_stds

    for c in cols:
        valid[f"z_{c}"] = valid_z[c]
        test[f"z_{c}"] = test_z[c]

    return train, valid, test


# ---------------------------
# 2) Altman Z-Score (چند نسخه)
# ---------------------------

class AltmanZ:
    """
    پیاده‌سازی نسخه‌های رایج Z-Score:
    - EM (Emerging Markets) — متداول برای بازارهای نوظهور
    - Z' (Private firms)
    - Z'' (Non-manufacturers)
    خروجی: مقدار Z به‌صورت پیوسته + امکان ارزیابی AUC/PR و طبقه‌بندی آستانه‌ای
    """
    def __init__(self, variant: str = "EM", mapping: Dict[str, str] = None):
        self.variant = variant
        self.m = mapping or {}

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = self.m
        TA = df[m["total_assets"]]
        X1 = df[m["working_capital"]] / TA
        X2 = df[m["retained_earnings"]] / TA
        X3 = df[m["ebit"]] / TA
        sales = df[m["sales"]]
        TL = df[m.get("total_liabilities", m["total_assets"]).split("#")[0]] if m.get("total_liabilities") else None
        MVE = df[m.get("market_value_equity", m.get("book_equity", ""))] if m.get("market_value_equity") or m.get("book_equity") else None

        if self.variant == "EM":
            # یکی از نسخه‌های رایج EM (برای بازارهای نوظهور)
            # Z = 3.25 + 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
            # X4 = Book Equity / Total Liabilities (یا MVE/TL بسته به منبع)
            if TL is None:
                raise ValueError("برای EM به total_liabilities نیاز است.")
            BE = df[m.get("book_equity", "")]
            X4 = BE / TL
            z = 3.25 + 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
        elif self.variant == "Zprime":
            # Z' برای شرکت‌های خصوصی تولیدی:
            # Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
            # X4 = Book Equity / Total Liabilities; X5 = Sales / Total Assets
            if TL is None:
                raise ValueError("برای Z' به total_liabilities نیاز است.")
            BE = df[m.get("book_equity", "")]
            X4 = BE / TL
            X5 = sales / TA
            z = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
        elif self.variant == "Zdoubleprime":
            # Z'' برای غیرتولیدی/خدمات:
            # Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
            if TL is None:
                raise ValueError("برای Z'' به total_liabilities نیاز است.")
            BE = df[m.get("book_equity", "")]
            X4 = BE / TL
            z = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
        else:
            raise ValueError("variant نامعتبر")
        return z


# ---------------------------
# 3) مدل‌ها (Registry)
# ---------------------------

class BaseModel:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_coef(self, feature_names: List[str]) -> Optional[pd.DataFrame]:
        return None


class LogisticL1(BaseModel):
    def __init__(self, C: float = 0.5, max_iter: int = 500, class_weight: Optional[Dict[int, float]] = None, random_state: int = 42):
        self.model = LogisticRegression(
            penalty="l1", solver="liblinear", C=C, max_iter=max_iter,
            class_weight=class_weight, random_state=random_state
        )
        self.calibrator: Optional[CalibratedClassifierCV] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, calibrate: bool = True, method: str = "isotonic") -> "LogisticL1":
        self.model.fit(X, y)
        if calibrate:
            self.calibrator = CalibratedClassifierCV(self.model, cv="prefit", method=method)
            self.calibrator.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)[:, 1]

    def get_coef(self, feature_names: List[str]) -> Optional[pd.DataFrame]:
        coefs = self.model.coef_.ravel()
        return pd.DataFrame({"feature": feature_names, "coef": coefs}).sort_values("coef", ascending=False)


class XGBoostModel(BaseModel):
    def __init__(self, params: Dict[str, Any]):
        if xgb is None:
            raise ImportError("xgboost نصب نیست.")
        self.model = xgb.XGBClassifier(**params)
        self.calibrator: Optional[CalibratedClassifierCV] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, calibrate: bool = True, method: str = "isotonic") -> "XGBoostModel":
        self.model.fit(X, y)
        if calibrate:
            self.calibrator = CalibratedClassifierCV(self.model, cv="prefit", method=method)
            self.calibrator.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)[:, 1]


def build_model(cfg: Config) -> BaseModel:
    if cfg.model_name == "logit_l1":
        return LogisticL1(C=cfg.C, max_iter=cfg.max_iter, class_weight=cfg.class_weight, random_state=cfg.random_state)
    elif cfg.model_name == "xgboost":
        return XGBoostModel(params=cfg.xgb_params)
    else:
        raise ValueError("مدل ناشناخته")


# ---------------------------
# 4) آستانه و هزینه
# ---------------------------

def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - (1 - fpr)
    idx = np.argmax(j)
    return float(thr[idx])


def f1_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.nanargmax(f1)
    # precision_recall_curve thr طولش یک کمتر از prec/rec است؛ ایمن‌سازی می‌کنیم
    return float(thr[max(0, min(idx, len(thr)-1))])


def cost_based_threshold(y_true: np.ndarray, y_prob: np.ndarray, c_fp: float, c_fn: float) -> float:
    # حد بهینه بیزی: tau* = C_FP / (C_FP + C_FN) اگر مجموع هزینه‌ها برابر و احتمال‌ها کالیبره باشند.
    # برای استحکام بیشتر، جست‌وجوی شبکه‌ای انجام می‌دهیم.
    taus = np.linspace(0.01, 0.99, 199)
    best_tau, best_cost = 0.5, np.inf
    for t in taus:
        y_hat = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        cost = c_fp*fp + c_fn*fn
        if cost < best_cost:
            best_cost = cost
            best_tau = t
    return float(best_tau)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, strategy: str = "youden", c_fp: float = 1.0, c_fn: float = 5.0) -> float:
    if strategy == "youden":
        return youden_threshold(y_true, y_prob)
    elif strategy == "f1":
        return f1_optimal_threshold(y_true, y_prob)
    elif strategy == "cost":
        return cost_based_threshold(y_true, y_prob, c_fp, c_fn)
    else:
        return 0.5


# ---------------------------
# 5) متریک‌ها، Bootstrap CI و DeLong
# ---------------------------

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    data = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = np.argsort(data)
    data_sorted = data[order]
    labels_sorted = labels[order]
    cum_pos = np.cumsum(labels_sorted) / (labels.sum() + 1e-9)
    cum_neg = np.cumsum(1 - labels_sorted) / (len(labels) - labels.sum() + 1e-9)
    ks = np.max(np.abs(cum_pos - cum_neg))
    return float(ks)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ks = ks_statistic(y_true, y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    spec = tn / (tn + fp + 1e-9)
    sens = tp / (tp + fn + 1e-9)
    return {
        "AUC": auc, "AUPRC": aupr, "Brier": brier, "KS": ks,
        "F1": f1, "Precision": prec, "Recall": rec, "Specificity": spec, "Sensitivity": sens,
        "Threshold": threshold,
    }


# DeLong implementation
# Adapted from common formulations; light-weight version

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_roc_variance(ground_truth, predictions):
    # ground_truth: 1/0, predictions: scores
    order = np.argsort(-predictions)
    predictions = predictions[order]
    ground_truth = ground_truth[order]
    distinct_value_indices = np.where(np.diff(predictions))[0]
    threshold_idxs = np.r_[distinct_value_indices, ground_truth.size - 1]

    tps = np.cumsum(ground_truth)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    P = np.sum(ground_truth)
    N = ground_truth.size - P

    fpr = fps / float(N)
    tpr = tps / float(P)

    v01 = tpr[1:] - tpr[:-1]
    v10 = fpr[1:] - fpr[:-1]

    # AUC
    auc = np.trapz(tpr, fpr)

    # Structural components
    sx = v10
    sy = v01
    v_xx = np.sum(sx * sx)
    v_yy = np.sum(sy * sy)

    return auc, v_xx / (P * N), v_yy / (P * N)


def delong_roc_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    auc1, var1_x, var1_y = delong_roc_variance(y_true, p1)
    auc2, var2_x, var2_y = delong_roc_variance(y_true, p2)
    var = var1_x + var1_y + var2_x + var2_y
    se = np.sqrt(var + 1e-12)
    z = (auc1 - auc2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(auc1 - auc2), float(p)


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 500, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats_list = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        stats_list.append(metric_fn(yt, yp))
    stats_arr = np.array(stats_list)
    return float(np.mean(stats_arr)), float(np.percentile(stats_arr, 2.5)), float(np.percentile(stats_arr, 97.5))


# ---------------------------
# 6) جریان اصلی
# ---------------------------

def run_pipeline(cfg: Config) -> Dict[str, Any]:
    print("CONFIG:\n", asdict(cfg))
    df = pd.read_csv(cfg.data_path)

    # فیلتر سطرهای ناقص برچسب
    df = df.dropna(subset=[cfg.label_col])
    # Split زمانی
    train, valid, test = temporal_split(df, cfg)

    # پیش‌پردازش: Winsorize + Impute + Standardize (درون‌سالی)
    for part_name, part in zip(["Train", "Valid", "Test"], [train, valid, test]):
        part[cfg.features] = part[cfg.features].apply(pd.to_numeric, errors="coerce")

    train = winsorize_by_group(train, cfg.features, cfg.group_standardize_cols, cfg.winsorize_q)
    valid = winsorize_by_group(valid, cfg.features, cfg.group_standardize_cols, cfg.winsorize_q)
    test  = winsorize_by_group(test,  cfg.features, cfg.group_standardize_cols, cfg.winsorize_q)

    train = impute_missing(train, cfg.features, strategy=cfg.impute_strategy, group_cols=cfg.group_standardize_cols)
    valid = impute_missing(valid, cfg.features, strategy=cfg.impute_strategy, group_cols=cfg.group_standardize_cols)
    test  = impute_missing(test,  cfg.features, strategy=cfg.impute_strategy, group_cols=cfg.group_standardize_cols)

    train, valid, test = standardize_by_group(train, valid, test, cfg.features, cfg.group_standardize_cols)

    z_features = [f"z_{c}" for c in cfg.features]

    # مدل
    model = build_model(cfg)
    X_tr, y_tr = train[z_features], train[cfg.label_col].astype(int).values
    model.fit(X_tr, y_tr, calibrate=cfg.calibrate, method=cfg.calibration_method)

    # پیش‌بینی
    X_va, y_va = valid[z_features], valid[cfg.label_col].astype(int).values
    X_te, y_te = test[z_features], test[cfg.label_col].astype(int).values

    p_tr = model.predict_proba(X_tr)
    p_va = model.predict_proba(X_va)
    p_te = model.predict_proba(X_te)

    # انتخاب آستانه روی Validation
    tau = select_threshold(y_va, p_va, strategy=cfg.threshold_strategy, c_fp=cfg.cost_fp, c_fn=cfg.cost_fn)

    # متریک‌ها
    metrics_tr = compute_metrics(y_tr, p_tr, tau)
    metrics_va = compute_metrics(y_va, p_va, tau)
    metrics_te = compute_metrics(y_te, p_te, tau)

    results = {
        "threshold": tau,
        "metrics_train": metrics_tr,
        "metrics_valid": metrics_va,
        "metrics_test": metrics_te,
        "coef": None,
    }

    coef_df = getattr(model, "get_coef", lambda _: None)(z_features)
    results["coef"] = coef_df

    # Altman Z: به‌صورت پیوسته + طبقه‌بندی آستانه‌ای کلاسیک (اختیاری)
    alt = AltmanZ(cfg.altman_variant, cfg.altman_mapping)
    for part_name, part, y, container in [
        ("Train", train, y_tr, results), ("Valid", valid, y_va, results), ("Test", test, y_te, results)
    ]:
        try:
            z = alt.compute(part)
            container[f"Altman_{part_name}_score"] = z.values
        except Exception as e:
            print(f"[Altman] {part_name}: {e}")

    # مقایسه AUC آلتمن vs مدل جدید روی Test (DeLong)
    if "Altman_Test_score" in results:
        z_test = results["Altman_Test_score"]
        delta_auc, p_val = delong_roc_test(y_te, p_te, z_test)
        results["delong_delta_auc"] = delta_auc
        results["delong_p_value"] = p_val

    # Bootstrap CI برای AUC/AUPRC روی Test
    mean_auc, lo_auc, hi_auc = bootstrap_ci(y_te, p_te, lambda yt, yp: roc_auc_score(yt, yp), n_boot=cfg.n_boot, seed=cfg.random_state)
    mean_pr, lo_pr, hi_pr = bootstrap_ci(y_te, p_te, lambda yt, yp: average_precision_score(yt, yp), n_boot=cfg.n_boot, seed=cfg.random_state)
    results["auc_ci"] = (mean_auc, lo_auc, hi_auc)
    results["auprc_ci"] = (mean_pr, lo_pr, hi_pr)

    # خروجی‌های خلاصه برای ذخیره
    summary_rows = []
    for split_name, m in [("Train", metrics_tr), ("Valid", metrics_va), ("Test", metrics_te)]:
        row = {"Split": split_name}
        row.update(m)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # ذخیره خروجی‌ها کنار فایل داده
    out_prefix = cfg.data_path.rsplit(".", 1)[0]
    summary_path = f"{out_prefix}_summary.csv"
    coef_path = f"{out_prefix}_logit_coef.csv"
    summary_df.to_csv(summary_path, index=False)
    if coef_df is not None:
        coef_df.to_csv(coef_path, index=False)

    print("Saved:", summary_path)
    if coef_df is not None:
        print("Saved:", coef_path)

    return results


# ---------------------------
# 7) اجرای نمونه با داده مصنوعی (برای تست اسکلت)
# ---------------------------

def _make_toy_data(n_firms: int = 60, years: List[int] = list(range(1392, 1404)), seed: int = 123) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for y in years:
        for i in range(n_firms):
            firm = f"F{i:03d}"
            # ویژگی‌های ساختگی با همبستگی معقول با ریسک
            base_risk = rng.beta(2, 8)
            fx = rng.normal(0, 1)
            cfo_assets = rng.normal(0.05 - base_risk*0.05, 0.03)
            ebit_assets = rng.normal(0.08 - base_risk*0.08, 0.04)
            st_debt_assets = rng.normal(0.2 + base_risk*0.3, 0.1)
            days_inv = rng.normal(120 + base_risk*80, 20)
            dso = rng.normal(90 + base_risk*60, 15)
            roa = ebit_assets
            features = {
                "current_ratio": 1.2 - base_risk*0.5 + rng.normal(0, 0.1),
                "quick_ratio": 0.9 - base_risk*0.4 + rng.normal(0, 0.1),
                "cash_ratio": 0.2 - base_risk*0.1 + rng.normal(0, 0.05),
                "debt_to_assets": 0.5 + base_risk*0.4 + rng.normal(0, 0.1),
                "debt_to_equity": 1.0 + base_risk*1.0 + rng.normal(0, 0.2),
                "st_debt_to_assets": st_debt_assets,
                "roa": roa,
                "roe": roa + rng.normal(0, 0.05),
                "ebit_to_assets": ebit_assets,
                "gross_margin": 0.4 - base_risk*0.1 + rng.normal(0, 0.05),
                "operating_margin": 0.2 - base_risk*0.08 + rng.normal(0, 0.04),
                "asset_turnover": 1.2 - base_risk*0.5 + rng.normal(0, 0.1),
                "inventory_turnover": 3.0 - base_risk*1.0 + rng.normal(0, 0.3),
                "days_inventory": days_inv,
                "dso": dso,
                "accruals_to_assets": 0.05 + base_risk*0.05 + rng.normal(0, 0.02),
                "cfo_to_assets": cfo_assets,
                "sales_growth": rng.normal(0.05, 0.1),
                "ebitda_growth": rng.normal(0.04, 0.1),
                "margin_cov": rng.uniform(0.05, 0.2),
                "import_dependency": rng.uniform(0.3, 0.8),
                "days_inventory_key_apis": days_inv + rng.normal(0, 10),
                "fx_exposure": rng.uniform(0.2, 0.9),
                "govt_sales_share": rng.uniform(0.2, 0.6),
                "insurance_receivables_delay": dso + rng.normal(0, 10),
                "log_assets": rng.normal(14, 1.0),
            }
            # اقلام آلتمن ساختگی
            total_assets = np.exp(rng.normal(15, 0.5))
            total_liabilities = total_assets * (0.4 + base_risk*0.4)
            working_capital = total_assets * (0.1 - base_risk*0.05)
            retained_earnings = total_assets * (0.15 - base_risk*0.1)
            ebit = total_assets * ebit_assets
            sales = total_assets * (1.2 - base_risk*0.5)
            equity_book = total_assets - total_liabilities
            mve = equity_book * (1.0 + rng.normal(0, 0.1))

            # برچسب: تابع لوجیت از چند ویژگی کلیدی + نویز
            logit = (
                1.5
                - 6.0 * cfo_assets
                - 4.0 * ebit_assets
                + 2.5 * st_debt_assets
                + 0.01 * (days_inv - 120)
                + 0.008 * (dso - 80)
            )
            p = 1 / (1 + np.exp(-logit))
            label = rng.binomial(1, min(max(p, 0.01), 0.8))

            row = {
                "firm_id": firm,
                "fiscal_year": y,
                "label": label,
                **features,
                "working_capital": working_capital,
                "retained_earnings": retained_earnings,
                "ebit": ebit,
                "sales": sales,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "equity_book": equity_book,
                "mve": mve,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def demo_with_toy(cfg: Config) -> Dict[str, Any]:
    df = _make_toy_data()
    path = "toy_pharma.csv"
    df.to_csv(path, index=False)
    cfg.data_path = path
    cfg.train_until = 1398
    cfg.valid_until = 1400
    return run_pipeline(cfg)


# ---------------------------
# 8) راه‌اندازی
# ---------------------------
if __name__ == "__main__":
    # مثال: اجرای دمو با داده مصنوعی
    cfg = Config(
        data_path="toy_pharma.csv",  # در دمو با demo_with_toy تولید می‌شود
        id_col="firm_id",
        time_col="fiscal_year",
        label_col="label",
        train_until=1398,
        valid_until=1400,
        model_name="xgboost",
        class_weight={0:1.0, 1:3.0},
        threshold_strategy="youden",
        altman_variant="EM",
    )
    results = demo_with_toy(cfg)
    print("\n=== SUMMARY (Test) ===")
    for k, v in results.get("metrics_test", {}).items():
        print(f"{k}: {v}")
    if results.get("coef") is not None:
        print("\nTop coefficients:")
        print(results["coef"].head(12))
    if "delong_p_value" in results:
        print(f"\nDeLong ΔAUC p-value: {results['delong_p_value']:.4f}")
