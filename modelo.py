import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. GENERACIÓN DEL DATASET
np.random.seed(100)
estaciones = ['Centro','Terminal','La Paz','Estacion Tren','Universidad','Mercado','Hospital','Parque']
horas = ['mañana','tarde','noche']
dias = ['laboral','fin_de_semana']
congestion_niveles = ['baja','media','alta']

tiempos = {
    ('Centro','Terminal'): 4, 
    ('Centro','La Paz'): 3, 
    ('Terminal','Universidad'): 6,
    ('La Paz','Estacion Tren'): 2, 
    ('Estacion Tren','Universidad'): 5, 
    ('Centro','Mercado'): 5,
    ('Mercado','Hospital'): 4, 
    ('Hospital','Parque'): 3, 
    ('Parque','Universidad'): 4,
    ('Terminal','Mercado'): 7,
}

registros = []
n_muestras = 300
for _ in range(n_muestras):
    orig = np.random.choice(estaciones)
    dest = np.random.choice([e for e in estaciones if e != orig])
    hora = np.random.choice(horas)
    dia = np.random.choice(dias)

    par = (orig, dest) if (orig, dest) in tiempos else (dest, orig)
    costo_base = tiempos.get(par, np.random.randint(5, 20))

    transbordos = np.random.randint(0, 4)
    congestion = np.random.choice(congestion_niveles)

    factor_cong = {'baja': 1.0, 'media': 1.3, 'alta': 1.7}[congestion]
    factor_hora = {'mañana': 1.4, 'tarde': 1.2, 'noche': 0.9}[hora]
    factor_dia = {'laboral': 1.2, 'fin_de_semana': 0.9}[dia]

    costo_ruta = round(costo_base * factor_cong * factor_hora * factor_dia, 1)

    
    es_optima = int(costo_ruta < 12 and transbordos <= 1 and congestion in ['baja', 'media'])

    registros.append({
        'estacion_origen': orig, 'estacion_destino': dest, 'hora_del_dia': hora,
        'dia_semana': dia, 'costo_ruta': costo_ruta, 'num_transbordos': transbordos,
        'congestion': congestion, 'ruta_optima': es_optima
    })

df = pd.DataFrame(registros)
print("DATASET GENERADO EXITOSAMENTE (ZIPAQUIRÁ)")
print("=" * 65)
print(df.head(10).to_string(index=False))
print(f"\nTotal registros: {len(df)} | Distribución clases (0=No Óptima, 1=Óptima):")
print(df['ruta_optima'].value_counts().to_string())


le_dict = {}
cols_categoricas = ['estacion_origen', 'estacion_destino', 'hora_del_dia', 'dia_semana', 'congestion']
df_enc = df.copy()

for col in cols_categoricas:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df_enc.drop(columns=['ruta_optima'])
y = df_enc['ruta_optima']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modelo_valen = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
modelo_valen.fit(X_train, y_train)
y_pred = modelo_valen.predict(X_test)
print("\n" + "=" * 65)
print("RENDIMIENTO")
print(f"Exactitud Global (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nReporte de Clasificación Metrológico:")
print(classification_report(y_test, y_pred, target_names=['No Óptima', 'Óptima']))
print("Matriz de Confusión Operativa:")
print(confusion_matrix(y_test, y_pred))


# GRÁFICO EN TERMINAL
print("\n" + "=" * 65)
print("IMPACTO DE CADA VARIABLE EN EL SISTEMA")
importancias = pd.Series(modelo_valen.feature_importances_, index=X.columns)
importancias_ord = importancias.sort_values(ascending=False)
for feat, imp in importancias_ord.items():
    barra = "" * int(imp * 40)
    print(f"  {feat:<22} {barra} {imp:.4f}")

# GRÁFICO DEL ÁRBOL 
fig, ax = plt.subplots(figsize=(18, 9), facecolor='#F8F9FA')
arbol_visual = plot_tree(
    modelo_valen,
    feature_names=list(X.columns),
    class_names=['Descartar (No Óptima)', 'Recomendar (Óptima)'],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    impurity=False,  
    proportion=True  
)
color_no_optima = '#E6CCB2'
color_optima = '#D8F3DC'

for nodo in arbol_visual:
    texto = nodo.get_text()
    if "Recomendar" in texto:
        nodo.set_bbox(dict(boxstyle="round,pad=0.4", facecolor=color_optima, edgecolor='#2D6A4F', linewidth=1.5))
    elif "Descartar" in texto:
        nodo.set_bbox(dict(boxstyle="round,pad=0.4", facecolor=color_no_optima, edgecolor='#7F5539', linewidth=1.5))
    else:
        nodo.set_bbox(dict(boxstyle="round,pad=0.4", facecolor='#EDF2F4', edgecolor='#2B2D42', linewidth=1.2))

plt.title(
    "MODELO PREDICTIVO SUPERVISADO - CONFIGURACIÓN ESTRATÉGICA DE RUTAS\nZipaquirá", 
    fontsize=14, fontweight='bold', color='#1D3557', pad=20
)

# GUARDAR IMAGEN
nombre_imagen = "arbol_supervisado.png"
plt.savefig(nombre_imagen, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("\n" + "-" * 65)
print("=" * 65)
plt.show()

# GUARDAR ARCHIVO
df.to_csv('dataset_zipaquira_rutas.csv', index=False)
print("\n" + "=" * 65)
print("Se ha guardado el archivo en tu carpeta.")
print("=" * 65)