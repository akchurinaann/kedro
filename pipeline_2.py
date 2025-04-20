from kedro.pipeline import Pipeline, node
from va_akchurina.pipelines.data_science.nodes import select_features_and_train

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
    func=select_features_and_train,
    inputs=["preprocessed_train", "y_train", "params:paramet"],
    outputs=["final_model", "model_metrics", "selected_features"],
    name="train_model_node"
),
        ]
    )