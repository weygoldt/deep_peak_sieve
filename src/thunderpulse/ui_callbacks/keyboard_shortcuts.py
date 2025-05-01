from dash import ClientsideFunction, Dash, Input, Output, State, dcc, html


def create_shortcuts(app):
    app.clientside_callback(
        """
            function(id) {
                document.addEventListener("keydown", function(event) {
                    if (event.ctrlKey) {
                        if (event.key == 'y') {
                            document.getElementById('bt_load_data').click()
                            event.stopPropogation()
                        }
                    }
                });
                return window.dash_clientside.no_update
            }
        """,
        Output("bt_load_data", "id"),
        Input("bt_load_data", "id"),
    )
