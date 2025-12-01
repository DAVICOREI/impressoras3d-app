import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Previsão de Sucesso na Impressão 3D", layout="centered")

@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")
    return joblib.load(model_path)

@st.cache_resource
def load_dataset(csv_path: str):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

MODEL_PATH = "modelo_impressoras3D.pkl"
CSV_PATH = "dataset_impressoras3D_12k.csv"

model = load_model(MODEL_PATH)
df_ref = load_dataset(CSV_PATH)

st.title("🔧 Impressões 3D – Previsão de Sucesso")
st.write(
    "Este aplicativo utiliza um modelo de **aprendizagem de máquina** treinado em 12.000 "
    "registros de impressões 3D para estimar a chance de **sucesso** de uma nova impressão."
)

# Opções categóricas a partir do dataset (se disponível)
def get_options(col, default_list):
    if df_ref is not None and col in df_ref.columns:
        vals = sorted(df_ref[col].dropna().unique().tolist())
        return vals if len(vals) > 0 else default_list
    return default_list

printer_model = st.selectbox(
    "Modelo da impressora",
    get_options("printer_model", ["Ender 3", "Ender 3 Pro", "Ender 3 V2"])
)

filament_material = st.selectbox(
    "Material do filamento",
    get_options("filament_material", ["PLA", "ABS", "PETG"])
)

filament_color = st.selectbox(
    "Cor do filamento",
    get_options("filament_color", ["preto", "branco", "azul", "vermelho"])
)

nozzle_diameter_mm = st.number_input("Diâmetro do bico (mm)", min_value=0.1, max_value=1.0, value=0.4, step=0.1)
nozzle_temp_c = st.number_input("Temperatura do bico (°C)", min_value=150, max_value=260, value=200, step=1)
bed_temp_c = st.number_input("Temperatura da mesa (°C)", min_value=0, max_value=120, value=60, step=1)
layer_height_mm = st.number_input("Altura de camada (mm)", min_value=0.05, max_value=0.5, value=0.2, step=0.01)
infill_percent = st.number_input("Infill (%)", min_value=0, max_value=100, value=20, step=1)
print_speed_mm_s = st.number_input("Velocidade de impressão (mm/s)", min_value=10, max_value=200, value=60, step=1)
part_volume_cm3 = st.number_input("Volume da peça (cm³)", min_value=0.0, max_value=2000.0, value=20.0, step=1.0)
print_time_hours = st.number_input("Tempo estimado de impressão (horas)", min_value=0.1, max_value=72.0, value=4.0, step=0.5)
ambient_temp_c = st.number_input("Temperatura ambiente (°C)", min_value=10, max_value=40, value=25, step=1)

support_used = st.selectbox("Suporte utilizado?", ["Sim", "Não"])
support_used_val = 1 if support_used == "Sim" else 0

st.markdown("---")

if st.button("Prever sucesso da impressão"):
    input_data = pd.DataFrame([{
        "printer_model": printer_model,
        "filament_material": filament_material,
        "filament_color": filament_color,
        "nozzle_diameter_mm": nozzle_diameter_mm,
        "nozzle_temp_c": nozzle_temp_c,
        "bed_temp_c": bed_temp_c,
        "layer_height_mm": layer_height_mm,
        "infill_percent": infill_percent,
        "print_speed_mm_s": print_speed_mm_s,
        "part_volume_cm3": part_volume_cm3,
        "print_time_hours": print_time_hours,
        "support_used": support_used_val,
        "ambient_temp_c": ambient_temp_c,
    }])

    try:
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.success(f"Alta chance de sucesso! Probabilidade estimada: {proba:.2%}")
        else:
            st.error(f"Risco maior de falha. Probabilidade de sucesso: {proba:.2%}")
    except Exception as e:
        st.error(f"Ocorreu um erro ao fazer a previsão: {e}")
