from dash import dcc


def create_layout_probe_graph():
    probe_graph = dcc.Graph(
        id="probe",
        responsive=True,
        # style={
        #     "height": "20vh",
        #     "width": "50%",
        # },
        config={"frameMargins": 0.0},
    )
    return probe_graph
