import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict]:

    # 1. Извлечение параметров
    preprocessing = params.get("paramet", {})
    if not preprocessing:
        preprocessing = {
            "target_column": "SalePrice",
            "categorical_columns": None,
            "numeric_columns": None,
            "outlier_threshold": 1.0,
            "feature_selection_estimators": 100,
            "random_state": 42,
            "n_features": 30
        }
    target_column = preprocessing.get("target_column")
    categorical_columns = preprocessing.get("categorical_columns")
    numeric_columns = preprocessing.get("numeric_columns")
    outlier_threshold = preprocessing.get("outlier_threshold", 1.0)
    feature_selection_estimators = preprocessing.get("feature_selection_estimators",100)
    random_state = preprocessing.get("random_state",42)
    n_features = preprocessing.get("n_features",30)

    # 2 Делаем индексами столбец айдишника
    train_df = train_df.set_index('Id')
    test_df = test_df.set_index('Id')

    # 3. Копирование данных для безопасности
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # 4. Автоматическое определение типов признаков
    if categorical_columns is None:
        categorical_columns = train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        categorical_columns = [col for col in categorical_columns if col != target_column]
    
    if numeric_columns is None:
        numeric_columns = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != target_column]
 

    # 6. Label Encoding для категориальных признаков
    encoders = {}

    if categorical_columns:
        for col in categorical_columns:
            # Объединяем train и test для согласованного кодирования
            combined = pd.concat([train_df[col], test_df[col]])
            le = LabelEncoder()
            le.fit(combined.astype(str))
            
            # Применяем к train и test
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
            encoders[col] = le

        # 7. Разделение train данных на признаки и таргет
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.copy()  # Тестовые данные уже без таргета

       # 8. Отбор признаков

    train_index = X_train.index
    test_index = X_test.index
    
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
        n_features_to_select=preprocessing.get("n_features", 30)
    )
    X_train_selected = rfe.fit_transform(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]
    X_train = pd.DataFrame(X_train_selected, 
                         columns=selected_features,
                         index=train_index)
    
    X_test = pd.DataFrame(rfe.transform(X_test),
                        columns=selected_features,
                        index=test_index)

    # 9. Удаление выбросов (только train)

    print("Количество строк до удаления вылетов:", len(X_train))
    threshold = 2.5
    z_scores = X_train.apply(zscore)
    outliers_mask = (np.abs(z_scores) > threshold).any(axis=1)
    X_train = X_train[~outliers_mask]
    y_train = y_train[~outliers_mask]
    print("Количество строк после удаления вылетов:", len(X_train))
    

    # 10. Формирование информации о признаках
    feature_columns = X_train.columns.tolist()
    features_info = {
        'feature_columns': feature_columns,
        'original_categorical': categorical_columns,
        'numeric_columns': numeric_columns,
        'target_column': target_column,
        'label_encoders': encoders
    }
    y_train = y_train.to_frame()
    
    return X_train, X_test, y_train, features_info