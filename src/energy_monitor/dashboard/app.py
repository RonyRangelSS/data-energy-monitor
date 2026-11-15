import os

import streamlit as st

from energy_monitor.dashboard.client import MonitorClient
from energy_monitor.dashboard.views import (
    alerts_frame,
    series_frame,
    servers_frame,
    total_forecast_frame,
)

TIMEZONE = os.environ.get("DISPLAY_TIMEZONE", "Australia/Melbourne")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
REFRESH_SECONDS = 5
PAGES = ("Visão geral", "Servidor", "Alertas", "Previsão")

st.set_page_config(page_title="Monitor de consumo energético", layout="wide")
client = MonitorClient(API_URL)

st.sidebar.title("Monitor de consumo")
st.sidebar.caption("Tempo quase real sobre reprodução acelerada do dataset histórico.")
scenario = st.sidebar.text_input("Cenário", value=os.environ.get("DEFAULT_SCENARIO", "demo"))
page = st.sidebar.radio("Página", PAGES)
tariff = st.sidebar.number_input(
    "Tarifa (R$/kWh)", min_value=0.01, value=float(os.environ.get("TARIFF_KWH", "0.80")), step=0.05
)
idle_threshold = st.sidebar.selectbox("Limiar de ociosidade (% de CPU)", [5.0, 10.0, 15.0], index=1)


@st.fragment(run_every=REFRESH_SECONDS)
def overview() -> None:
    indicators = client.indicators(scenario, tariff, idle_threshold)["total"]
    alerts = client.alerts(scenario, status="open")
    columns = st.columns(4)
    columns[0].metric("Energia (kWh)", f"{indicators['energy_kwh']:.1f}")
    columns[1].metric("Custo associado (R$)", f"{indicators['associated_cost']:.2f}")
    columns[2].metric("Energia em ociosidade", f"{100 * indicators['idle_energy_share']:.1f}%")
    columns[3].metric("Alertas abertos", len(alerts))
    st.subheader("Servidores")
    st.dataframe(servers_frame(client.servers(scenario), TIMEZONE), use_container_width=True)


@st.fragment(run_every=REFRESH_SECONDS)
def server_page(host_id: str) -> None:
    series = series_frame(client.series(scenario, host_id), TIMEZONE)
    st.line_chart(series, y_label="Potência (W)")
    st.caption("Real × esperado para a carga atual × previsto.")


@st.fragment(run_every=REFRESH_SECONDS)
def alerts_page() -> None:
    st.dataframe(alerts_frame(client.alerts(scenario), TIMEZONE), use_container_width=True)


@st.fragment(run_every=REFRESH_SECONDS)
def forecast_page() -> None:
    forecasts = total_forecast_frame(client.forecasts(scenario, limit=2000), TIMEZONE)
    st.line_chart(forecasts, y_label="Potência total prevista (W)")
    st.caption("Soma das previsões dos servidores com histórico suficiente.")


st.title(page)
if page == PAGES[0]:
    overview()
elif page == PAGES[1]:
    hosts = [item["host_id"] for item in client.servers(scenario)]
    if hosts:
        server_page(st.selectbox("Servidor", hosts))
    else:
        st.info("Nenhuma leitura recebida neste cenário.")
elif page == PAGES[2]:
    alerts_page()
else:
    forecast_page()
