import plotly.graph_objs as go


class HistoriesStorage:
    """
    Can store history of many models fitting
    """
    def __init__(self):
        self._histories = []

    def store_history(self, history):
        self._histories.append(history)

    def get_histories(self):
        return self._histories


def plot_model_histories(histories_storage: HistoriesStorage, metric: str, save=False):
    """
    Make a plot from histories and given metric
    """
    fig = go.Figure()
    histories = histories_storage.get_histories()

    for fold_no in range(0, len(histories)):
        fold_values = histories[fold_no].history[metric]
        epochs = [i for i in range(1, len(fold_values) + 1)]
        fig.add_trace(go.Scatter(x=epochs,
                                 y=fold_values,
                                 name=f"Fold {fold_no + 1}",
                                 hovertemplate='Epoch: %{x}, Value: %{y:.4f}',
                                 showlegend=True)
                      )
    fig.update_layout(xaxis_title="Epoch",
                      yaxis_title="Accuracy",
                      title={
                          'text': f"Model {metric} statistics",
                          'font': {'size': 18}},
                      showlegend=True,
                      hoverlabel_align='right', )
    if save:
        fig.write_html(f'model_{metric}.html')
    fig.show()
