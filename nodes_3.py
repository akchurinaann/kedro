import pandas as pd
def make_predictions(
    final_model, 
    selected_features: list,
    X_test: pd.DataFrame
) -> pd.DataFrame:
   
    # Проверяем наличие столбца Id
    assert 'Id' in X_test.columns, "Входные данные должны содержать столбец 'Id'"
    
    # Создаем предсказания
    predictions = final_model.predict(X_test[selected_features])
    
    # Создаем результат с ID и предсказаниями
    result = pd.DataFrame({
        'Id': X_test['Id'].values,  # Берем значения столбца Id
        'PredictedPrice': predictions
    })
    
    return result