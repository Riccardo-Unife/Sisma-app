from flask import Flask, render_template, request, send_file
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.integrate import quad, cumulative_trapezoid
import io

app = Flask(__name__)

# Variabili globali per salvataggio dati
latest_data = {}

def generate_graphs(params):
    global latest_data

    # Parametri da input utente
    massa = params['massa']
    t_max = params['t_max']
    dt = params['dt']
    a1, b1, c1, m1, n1 = params['a1'], params['b1'], params['c1'], params['m1'], params['n1']
    a2, b2, c2, m2, n2 = params['a2'], params['b2'], params['c2'], params['m2'], params['n2']

    t_max = t_max / 2
    t = np.arange(-t_max, t_max, dt)
    t_real = np.arange(0, 2 * t_max, dt)

    m2 = 1 / m2
    n2 = 1 / n2

    lim_g = 270 / (massa + 4.5) * 0.0981
    lim_v = 500
    lim_s = 50

    f_acc = (m1 - n1) * np.exp(-(np.abs(((t - b1) ** 2) / (2 * a1 ** 2))) ** c1) + n1

    def f(t):
        return (m2 - n2) * np.exp(-(np.abs(((t - b2) ** 2) / (2 * a2 ** 2))) ** c2) + n2

    integrale = np.array([quad(f, 0, τ)[0] for τ in t])
    a_g = f_acc * np.sin(2 * np.pi * integrale)
    a_cms2 = a_g * 981
    v_cms = cumulative_trapezoid(a_cms2, t, initial=0)
    s_cm = cumulative_trapezoid(v_cms, t, initial=0)

    idx_zero = np.argmin(np.abs(t - 0))
    s_cm_at_t0 = s_cm[idx_zero]
    s_cm = s_cm - s_cm_at_t0

    # Salva dati per il download
    latest_data = {
        "t_real": t_real,
        "a_g": a_g,
        "v_cms": v_cms,
        "s_cm": s_cm
    }

    # GRAFICI
    graphs = []

    # Accelerazione
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_real, y=f_acc, mode='lines', line=dict(dash='dash', color='gray'), opacity=0.3, showlegend=False))
    fig.add_trace(go.Scatter(x=t_real, y=-f_acc, mode='lines', line=dict(dash='dash', color='gray'), opacity=0.3, showlegend=False))
    fig.add_trace(go.Scatter(x=t_real, y=a_g, mode='lines', name='sisma [g]', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(x=t_real, y=[lim_g]*len(t), mode='lines', line=dict(dash='dash', color='gray'), opacity=0.7, showlegend=False))
    fig.add_trace(go.Scatter(x=t_real, y=[-lim_g]*len(t), mode='lines', line=dict(dash='dash', color='gray'), opacity=0.7, showlegend=False))
    fig.update_layout(title="Accelerazione", xaxis_title="tempo [s]", yaxis_title="acc [g]", template="plotly_dark")
    graphs.append(pio.to_html(fig, full_html=False))

    # Velocità
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_real, y=v_cms, mode='lines', name='v [cm/s]', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(x=t_real, y=[lim_v]*len(t), mode='lines', line=dict(dash='dash', color='gray'), opacity=0.7, showlegend=False))
    fig.add_trace(go.Scatter(x=t_real, y=[-lim_v]*len(t), mode='lines', line=dict(dash='dash', color='gray'), opacity=0.7, showlegend=False))
    fig.update_layout(title="Velocità", xaxis_title="tempo [s]", yaxis_title="vel [cm/s]", template="plotly_dark")
    graphs.append(pio.to_html(fig, full_html=False))

    # Spostamento
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_real, y=s_cm, mode='lines', name='s [cm]', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(x=t_real, y=[lim_s]*len(t), mode='lines', line=dict(dash='dash', color='gray'), opacity=0.7, showlegend=False))
    fig.add_trace(go.Scatter(x=t_real, y=[-lim_s]*len(t), mode='lines', line=dict(dash='dash', color='gray'), opacity=0.7, showlegend=False))
    fig.update_layout(title="Spostamento", xaxis_title="tempo [s]", yaxis_title="disp [cm]", template="plotly_dark")
    graphs.append(pio.to_html(fig, full_html=False))

    return graphs


@app.route('/', methods=['GET', 'POST'])
def index():
    default_values = {
        'massa': 5, 't_max': 20, 'dt': 0.005,
        'a1': 3.5, 'b1': 0, 'c1': 1.5, 'm1': 1.5, 'n1': 0,
        'a2': 5, 'b2': 0, 'c2': 3, 'm2': 1, 'n2': 0.2
    }

    graphs = []

    if request.method == 'POST':
        try:
            values = {k: float(request.form[k]) for k in default_values.keys()}
            graphs = generate_graphs(values)
            return render_template("index.html", params=values, graphs=graphs)
        except ValueError:
            return render_template("index.html", params=default_values, graphs=[], error="Errore nei parametri.")
    else:
        return render_template("index.html", params=default_values, graphs=[])

@app.route('/download')
def download():
    global latest_data
    if not latest_data:
        return "Nessun dato disponibile."

    buf = io.StringIO()
    buf.write("t_real\t a_g\t v_cms\t s_cm\n")
    for i in range(len(latest_data["t_real"])):
        buf.write(f"{latest_data['t_real'][i]:.5f}\t{latest_data['a_g'][i]:.5f}\t{latest_data['v_cms'][i]:.5f}\t{latest_data['s_cm'][i]:.5f}\n")
    buf.seek(0)

    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype='text/plain', as_attachment=True, download_name='output_dati.txt')


if __name__ == '__main__':
    app.run(debug=True)
