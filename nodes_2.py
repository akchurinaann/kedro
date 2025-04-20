import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def select_features_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict
) -> tuple:
   
    # 1. Подготовка данных
    y_train = np.ravel(y_train)

    # 2. Извлечение параметров
    model = params.get("paramet", {})
    if not model:
        model = {
            "target_column": "SalePrice",
            "categorical_columns": None,
            "numeric_columns": None,
            "outlier_threshold": 1.0,
            "feature_selection_estimators": 100,
            "random_state": 42,
            "n_features_1": 20,
            "n_estimators": 300,
            "cv_folds": 7
        }
    target_column = model.get("target_column")
    categorical_columns = model.get("categorical_columns")
    numeric_columns = model.get("numeric_columns")
    outlier_threshold = model.get("outlier_threshold", 1.0)
    feature_selection_estimators = model.get("feature_selection_estimators",100)
    random_state = model.get("random_state",42)
    n_features_1 = model.get("n_features",20)
    n_estimators = model.get("n_features",300)
    cv_folds = model.get("cv_folds",7)
    

    # 3. Отбор признаков
    model_for_selection = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
    rfe = RFE(
        estimator=model_for_selection,
        n_features_to_select=model.get("n_features_1", 20)
    )
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_].tolist()
    
    # 4. Обучение модели
    final_model = RandomForestRegressor(
    n_estimators=model.get("n_estimators", 300),
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=model.get("random_state", 42),
    n_jobs=-1
)
    final_model.fit(X_train[selected_features], y_train)
    
    # 5. Расчет метрик
    y_pred = final_model.predict(X_train[selected_features])
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_train, y_pred)),
        'R2': r2_score(y_train, y_pred),
        'CV_RMSE': np.mean(np.sqrt(-cross_val_score(
            final_model,
            X_train[selected_features],
            y_train,
            cv=model.get("cv_folds", 7),
            scoring='neg_mean_squared_error',
            n_jobs=-1
        ))),
        'selected_features': selected_features
    }
    
    return final_model, metrics, selected_features