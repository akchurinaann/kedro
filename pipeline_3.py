from kedro.pipeline import Pipeline, node
from va_akchurina.pipelines.reporting.nodes import make_predictions

def create_pipeline(**kwargs):
    return Pipeline([
  node(
                func=make_predictions,
                inputs=["final_model", "selected_features", "preprocessed_test"],
                outputs="predictions",
                name="make_predictions_node"
            )
    ])