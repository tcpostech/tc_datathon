"""
Treinamento e salvamento do modelo.
Pipeline de treino com preprocessing, feature_engineering, XGBoost e MLflow.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from src.evaluate import compute_metrics
from src.feature_engineering import criar_atributos_derivados
from src.preprocessing import preparar_dados
from src.utils import encontrar_coluna

logger = logging.getLogger(__name__)

# Nome da coluna alvo (conforme utils e notebook)
TARGET_COL = "Defasagem"
MLFLOW_EXPERIMENT_NAME = "pede_xgboost"
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MLRUNS_DIR = Path("mlruns")


def _get_target_column(df: pd.DataFrame) -> str:
    """Retorna o nome real da coluna alvo no DataFrame."""
    found = encontrar_coluna(TARGET_COL, list(df.columns))
    if found is None:
        raise ValueError(f"Coluna alvo '{TARGET_COL}' não encontrada. Colunas: {list(df.columns)}")
    return found


def _prepare_features_target(
        df: pd.DataFrame,
        target_col: str,
        fill_strategy: str = "median",
) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    """Separa features e target; preenche faltantes; retorna (X, y, feature_names)."""
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Apenas numéricos
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("Nenhuma coluna numérica disponível para treino.")

    feature_names = list(X.columns)

    if fill_strategy == "median":
        X = X.fillna(X.median())
    else:
        X = X.fillna(0)

    # Remove linhas onde o target está faltando
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(float if y.dtype in (np.float64, np.float32) else int)

    return X, y, feature_names


def _infer_task(y: pd.Series) -> str:
    """Inferir se é classificação (binária/multiclasse) ou regressão."""
    n_unique = y.nunique()
    if n_unique <= 2:
        return "classification"
    if n_unique <= 20:
        return "classification"  # multiclasse
    return "regression"


def train(
        save_path: Optional[Path] = None,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        xgb_params: Optional[dict] = None,
) -> Optional[Any]:
    """
    Treina o modelo a partir dos dados PEDE 2024, registra no MLflow e salva em save_path.
    """
    mlflow.autolog()

    df = preparar_dados()
    df = criar_atributos_derivados(df)
    logger.info("Treino: %s linhas carregadas.", len(df))

    target_col = _get_target_column(df)
    X, y, feature_names = _prepare_features_target(df, target_col)
    task = _infer_task(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task == "classification" and y.nunique() <= 10 else None
    )

    # Parâmetros XGBoost
    default_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "eval_metric": "auc" if task == "classification" else "rmse",
    }
    params = {**default_params, **(xgb_params or {})}
    start_train = time.time()

    if task == "classification":
        n_class = int(y.nunique())
        if n_class == 2:
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBClassifier(**params, num_class=n_class)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        y_score = model.predict_proba(X_val)[:, 1] if n_class == 2 else None
    else:
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        y_score = None

    train_time = time.time() - start_train
    metrics = compute_metrics(y_val.values, y_pred, y_score=y_score, task=task)

    # MLflow
    exp_name = experiment_name or MLFLOW_EXPERIMENT_NAME
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("task", task)
        mlflow.log_param("feature_count", len(feature_names))
        mlflow.log_param("feature_names", str(feature_names))
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metrics(metrics)

        mlflow.xgboost.log_model(model, artifact_path="model")

        precision = precision_score(y_val, y_pred, pos_label=1)
        recall = recall_score(y_val, y_pred, pos_label=1)
        f1 = f1_score(y_val, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_val, y_score)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # gráficos
        # matriz de confusão
        ConfusionMatrixDisplay.from_estimator(model, X_val, y_val)
        mlflow.log_figure(plt.gcf(), 'rf_confusion_matrix.png')
        plt.close()

        # roc
        RocCurveDisplay.from_estimator(model, X_val, y_val)
        mlflow.log_figure(plt.gcf(), 'rf_roc_curve.png')
        plt.close()

        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run_id: %s", run_id)
        print(classification_report(y_val, y_pred, digits=3))

    # Salvar modelo e metadados para a API
    out_dir = Path(save_path) if save_path else DEFAULT_MODEL_DIR
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.json"
    model.save_model(str(model_path))
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False)
    with open(out_dir / "task.json", "w", encoding="utf-8") as f:
        json.dump({"task": task}, f)

    logger.info("Modelo salvo em %s", model_path)
    return model
