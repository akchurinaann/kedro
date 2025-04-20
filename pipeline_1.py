from kedro.pipeline import Pipeline, node, pipeline
from va_akchurina.pipelines.data_processing.nodes import preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    """Создает пайплайн предобработки данных с One-Hot Encoding"""
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=[
                    "raw_train_data",
                    "raw_test_data",
                    "params:paramet",  # Параметры предобработки
                ],
                outputs=[
                    "preprocessed_train",
                    "preprocessed_test",
                    "y_train",
                    "features_info"
                ],
                name="preprocess_data_node",
                tags=["preprocessing"]
            )
        ]
    )