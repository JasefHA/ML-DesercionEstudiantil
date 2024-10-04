# %% [markdown]
# # 1. CARGO DATAFRAMES BASE

# %%
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

ruta_export = r"C:\Users\MSII7\Documents\Jasef\Desercion\export"
ruta_output = r"C:\Users\MSII7\Documents\Jasef\Desercion\output"
ruta_sources = r"C:\Users\MSII7\Documents\Jasef\Desercion\sources"

df_matriculados_2024_1 = pd.read_csv(os.path.join(ruta_export,'Matriculados_2024_1.csv'))
df_matriculados_2023_2 = pd.read_csv(os.path.join(ruta_export,'Matriculados_2023_2.csv'))
df_matriculados_2023_1 = pd.read_csv(os.path.join(ruta_export,'Matriculados_2023_1.csv'))
df_matriculados_2022_2 = pd.read_csv(os.path.join(ruta_export,'Matriculados_2022_2.csv'))
df_matriculados_2022_1 = pd.read_csv(os.path.join(ruta_export,'Matriculados_2022_1.csv'))


df_desertores_2023_2 = pd.read_csv(os.path.join(ruta_export,'Desertores_2023_2.csv'))

df_variables = pd.read_csv(os.path.join(ruta_export,'DataMaestra_Estudiante.csv'))

df_notas_2023_2 = pd.read_csv(os.path.join(ruta_sources,'DATA_DETALLE_NOTAS-2023-2.csv'))

df_colegio_procedencia = pd.read_csv(
    os.path.join(ruta_sources, 'DATA_COLEGIO_PROCEDENCIA.csv'),
    header=None,  # Indica que el archivo no tiene cabeceras
    names=['IdAlumno', 'Colegio', 'TipoColegio']  # Especificar las cabeceras manualmente
)
df_colegio_procedencia = df_colegio_procedencia.drop_duplicates(subset='IdAlumno')

df_pagos_2023 = pd.read_csv(os.path.join(ruta_sources,'pagos cachimbos 2023-2.csv'))


nota_aprobatoria = 13
notas_desaprobadas = 7
periodo = '2023-2'

# %% [markdown]
# # CALCULO VARIABLES NOTAS

# %%
# Agrupar por IdAlumno y contar los registros de parciales y finales desaprobados
df_cant_desaprobado_agrupado = df_notas_2023_2.groupby('IdAlumno').agg(
    CantParcialesDesaprobados=(
        'Actividad', 
        lambda x: ((x == 'EVALUACION PARCIAL') & (df_notas_2023_2.loc[x.index, 'NotaActiv'] < nota_aprobatoria)).sum()
    ),
    CantFinalesDesaprobados=(
        'Actividad', 
        lambda x: ((x == 'EVALUACION FINAL') & (df_notas_2023_2.loc[x.index, 'NotaActiv'] < nota_aprobatoria)).sum()
    ),
    # Columna auxiliar para contar las evaluaciones diferentes de PARCIAL y FINAL desaprobadas
    CantNotasDesaprobadasNoParcNoFin=(
        'Actividad', 
        lambda x: ((~x.isin(['EVALUACION PARCIAL', 'EVALUACION FINAL', 'EVALUACION DIAGNOSTICA'])) & (df_notas_2023_2.loc[x.index, 'NotaActiv'] < nota_aprobatoria)).sum()
    )
).reset_index()

# Crear la columna 'ExcedeNotasDesaprobadasNoParcNoFin'
df_cant_desaprobado_agrupado['ExcedeNotasDesaprobadasNoParcNoFin'] = df_cant_desaprobado_agrupado['CantNotasDesaprobadasNoParcNoFin'].apply(
    lambda x: 'SI' if x >= notas_desaprobadas else 'NO'
)

# Eliminar la columna 'CantNotasDesaprobadasNoParcNoFin' 
df_cant_desaprobado_agrupado.drop(columns=['CantNotasDesaprobadasNoParcNoFin'], inplace=True)

# Calcular la nota final para cada curso por alumno
df_notas_2023_2['NotaFinalCurso'] = np.where(
    df_notas_2023_2['Actividad'] != 'EVALUACION DIAGNOSTICA',
    (df_notas_2023_2['Peso'] * df_notas_2023_2['NotaActiv']) / 100,
    np.nan  # Ignoramos las actividades diagnósticas
)

# Agrupar por IdAlumno y Curso para obtener la nota final por curso
df_notas_finales_por_curso = df_notas_2023_2.groupby(['IdAlumno', 'Curso']).agg(
    NotaFinalCurso=('NotaFinalCurso', 'sum')  # Sumar las notas ponderadas por curso
).reset_index()

# Redondeo las notas a su entero más cercano para el posterior conteo 
df_notas_finales_por_curso['NotaFinalCurso'] = df_notas_finales_por_curso['NotaFinalCurso'].round()

# Agrupar por IdAlumno y contar los cursos desaprobados
df_cursos_desaprobados = df_notas_finales_por_curso.groupby('IdAlumno').agg(
    CantCursosDesaprobados=(
        'NotaFinalCurso', 
        lambda x: (x < nota_aprobatoria).sum()
    ),
    CantCursos=(
        'NotaFinalCurso', 
        'count'  # Cuenta la cantidad de cursos
    )
).reset_index()

# Creamos la columna ExcedeMitadCursosDesaprobados
df_cursos_desaprobados['ExcedeMitadCursosDesaprobados'] = df_cursos_desaprobados.apply(
    lambda row: 'SI' if (row['CantCursosDesaprobados'] / row['CantCursos']) >= 0.5 else 'NO', axis=1
)

# Eliminar la columna 'CantNotasDesaprobadasNoParcNoFin' 
df_cursos_desaprobados.drop(columns=['CantCursosDesaprobados','CantCursos'], inplace=True)

# Unir este resultado con el DataFrame anterior
df_cant_desaprobado_agrupado = df_cant_desaprobado_agrupado.merge(
    df_cursos_desaprobados, 
    on='IdAlumno', 
    how='left'
)

df_cant_desaprobado_agrupado = df_cant_desaprobado_agrupado.merge(
    df_notas_2023_2[['IdAlumno', 'ProgramaAlu']].drop_duplicates(),  # Tomar IdAlumno y ProgramaAlu sin duplicados
    on='IdAlumno',
    how='left'
)

# Visualizar el DataFrame final
print(df_cant_desaprobado_agrupado.head())

# %% [markdown]
# # CALCULO DE VARIABLES DE PAGO

# %%
df_pagos_2023['FECHA_VENCIMIENTO'] = pd.to_datetime(df_pagos_2023['FECHA_VENCIMIENTO'])
df_pagos_2023['FECHA_TRANSACCION'] = pd.to_datetime(df_pagos_2023['FECHA_TRANSACCION'])

# Paso 1: Agrupar por IdAlumno y contar la cantidad de pagos (CantArmadas)
df_agrupado = df_pagos_2023.groupby('IdAlumno').agg(
    CantArmadas=('IdAlumno', 'size')
).reset_index()

# Paso 2: Calcular CantArmadasRetraso7dias (transacciones con retraso mayor a 7 días)
df_pagos_2023['Retraso'] = (df_pagos_2023['FECHA_TRANSACCION'] - df_pagos_2023['FECHA_VENCIMIENTO']).dt.days
df_pagos_2023['RetrasoMayor7Dias'] = df_pagos_2023['Retraso'] > 7

# Agrupar por IdAlumno y contar las transacciones con retraso mayor a 7 días
df_retraso = df_pagos_2023.groupby('IdAlumno').agg(
    CantArmadasRetraso7dias=('RetrasoMayor7Dias', 'sum')
).reset_index()

# Paso 3: Unir ambos DataFrames (df_agrupado y df_retraso)
df_resultado_pagos = pd.merge(df_agrupado, df_retraso, on='IdAlumno', how='left')

# Paso 4: Crear el nuevo campo con la lógica Si/No
df_resultado_pagos['ExcedePagosAtrasados'] = df_resultado_pagos.apply(
    lambda row: 1 if row['CantArmadasRetraso7dias'] >= (row['CantArmadas'] / 2) else 0,
    axis=1
)

print(df_resultado_pagos.head())

# %% [markdown]
# # IDENTIFICO LOS ESTUDIANTES A PREDECIR

# %%
# Unir todos los periodos anteriores en un solo dataframe
df_matriculados_anteriores = pd.concat([df_matriculados_2023_1, df_matriculados_2022_2, df_matriculados_2022_1])

# %%
df_nuevos_matriculados_2023_2 = df_matriculados_2023_2[~df_matriculados_2023_2['IdAlumno'].isin(df_matriculados_anteriores['IdAlumno'])]

# %%
# Filtrar el DataFrame para encontrar el registro específico
resultado = df_nuevos_matriculados_2023_2[df_nuevos_matriculados_2023_2['IdAlumno'] == 100118394]

# Verifica el resultado
print(resultado)

# %%
print(df_nuevos_matriculados_2023_2.shape)

# %%
print(df_variables.isnull().sum())

# %%
#Variables CRM
df = pd.merge(df_nuevos_matriculados_2023_2, df_variables, on='IdAlumno', how='left')

#Variables Notas
df = pd.merge(df, df_cant_desaprobado_agrupado, on='IdAlumno', how='left')

#Colegio Procedencia
df = pd.merge(df, df_colegio_procedencia, on='IdAlumno', how='left')

#Variables Pagos
df = pd.merge(df, df_resultado_pagos, on='IdAlumno', how='left')

# columna objetivo
df['Desercion'] = df['IdAlumno'].isin(df_desertores_2023_2['IdAlumno']).astype(int)

print(df.head())

# %%
print(df.shape)

# %% [markdown]
# ### RELLENO LOS QUE NO TIENEN GENERO
# 
# 

# %%
# Calculamos la cuenta de cada género y de los valores nulos
conteo_generos = df['Genero'].value_counts(dropna=False)
count_f = conteo_generos.get('F', 0)  # Cantidad de F
count_m = conteo_generos.get('M', 0)  # Cantidad de M
total_blancos = conteo_generos.get(np.nan, 0)  # Cantidad de valores nulos

# Ahora, calculamos la distribución equitativa
if total_blancos > 0:
    # Proporciones de cada género
    proporciones_f = count_f / (count_f + count_m)
    proporciones_m = count_m / (count_f + count_m)

    # Cálculo de cuántos géneros se asignarán a los valores nulos
    asignacion_f = int(total_blancos * proporciones_f)
    asignacion_m = total_blancos - asignacion_f  # Lo que queda se asigna a M

    # Rellenar los valores nulos en el DataFrame
    df.loc[df['Genero'].isna(), 'Genero'] = ['F'] * asignacion_f + ['M'] * asignacion_m


print(df.isnull().sum())

# %% [markdown]
# ### RELLENO LOS QUE NO TIENEN FECHA DE NACIMIENTO

# %%
# Contar cuántos valores nulos hay en FechaNacimiento
n = df['FechaNacimiento'].isnull().sum()

# Función para generar fechas aleatorias en el formato 'YYYY-MM-DD'
def generar_fecha_aleatoria(n):
    fechas = []
    for _ in range(n):
        anio = np.random.choice([2006, 2007])  # Años para tener 16 o 17 años en 2023
        mes = np.random.randint(1, 13)  # Mes de 1 a 12
        dia = np.random.randint(1, 29)  # Día de 1 a 28 (para simplificar)
        fecha = f"{anio}-{mes:02d}-{dia:02d}"  # Formato 'YYYY-MM-DD'
        fechas.append(fecha)
    return fechas

# Generar fechas aleatorias y rellenar los nulos
fechas_aleatorias = generar_fecha_aleatoria(n)
df.loc[df['FechaNacimiento'].isnull(), 'FechaNacimiento'] = fechas_aleatorias

# Obtener el año de nacimiento de los nulos en FechaNacimiento
df.loc[df['AnioNacimiento'].isnull(), 'AnioNacimiento'] = df['FechaNacimiento'].str.split('-').str[0].astype(int)

#print(df)
print(df.isnull().sum())


# %% [markdown]
# ### RELLENO LOS QUE NO TIENEN DISTRITO

# %%
# Lista de distritos para rellenar
distritos = ["ATE", "VILLA EL SALVADOR", "COMAS", "CHORRILLOS", "SAN MARTÍN DE PORRES"]

# Contar cuántos valores nulos hay en el campo 'Distrito'
nulos_distrito = df['Distrito'].isnull().sum()

# Calcular cuántos registros de cada distrito se necesitan
if nulos_distrito > 0:
    # Calcular cuántos distritos asignar a cada uno
    asignaciones = [nulos_distrito // len(distritos)] * len(distritos)
    
    # Distribuir los restantes de manera aleatoria entre los distritos
    for i in range(nulos_distrito % len(distritos)):
        asignaciones[i] += 1

    # Crear una lista con los distritos asignados
    distritos_asignados = []
    for distrito, cantidad in zip(distritos, asignaciones):
        distritos_asignados.extend([distrito] * cantidad)

    # Mezclar aleatoriamente la lista para distribuir uniformemente
    np.random.shuffle(distritos_asignados)

    # Rellenar los valores nulos en el DataFrame
    df.loc[df['Distrito'].isnull(), 'Distrito'] = distritos_asignados

# Mostrar el DataFrame resultante
print(df.isnull().sum())

# %% [markdown]
# ### RELLENO LOS QUE NO TIENEN PROVINCIA NI DEPARTAMENTO

# %%
df['Provincia'] = df['Provincia'].fillna('LIMA')
df['Departamento'] = df['Departamento'].fillna('LIMA')

print(df.isnull().sum())

# %% [markdown]
# ### CREO CAMPO EDAD

# %%
df['Edad'] = df['Anio'] - df['AnioNacimiento']
print(df.isnull().sum())

# %%
# Definir las condiciones y sus correspondientes etiquetas
condiciones = [
    (df['Edad'] <= 18),
    (df['Edad'] > 18) & (df['Edad'] < 25),
    (df['Edad'] >= 25)
]

etiquetas = ['MenosDe18', 'Entre18y25', 'Mayor25']

# Crear la nueva columna 'GrupoEdad' con las etiquetas según las condiciones, usando un valor por defecto como ''
df['GrupoEdad'] = np.select(condiciones, etiquetas, default='')

# Verificar los resultados
print(df[['Edad', 'GrupoEdad']].head())

# %% [markdown]
# ### RELLENO EL FAMILIAR RESPONSABLE

# %%
# Filas donde FamiliarResponsable es nulo
mask = df['FamiliarResponsable'].isnull()

# Aplicar la condición solo a esas filas
df.loc[mask, 'FamiliarResponsable'] = df.loc[mask, 'Edad'].apply(lambda x: 'Apoderado' if x <= 18 else 'Alumno')

print(df.isnull().sum())

# %% [markdown]
# ### Relleno el tipo de colegio de procedencia

# %%
df['TipoColegio'] = df_colegio_procedencia['TipoColegio'].replace({'RELG': 'PRIV', 'OTHR': 'PUBL'}).fillna('PUBL')

print(df.isnull().sum())

# %%
print(df.shape)
df.to_csv('cachimbos.csv')

# %%
variables_modelo = ['Edad', 'Genero', 'Distrito', 'Provincia', 'Departamento', 'FamiliarResponsable', 'CantParcialesDesaprobados', 'CantFinalesDesaprobados','ExcedeNotasDesaprobadasNoParcNoFin','ExcedeMitadCursosDesaprobados','GrupoEdad','ExcedePagosAtrasados','TipoColegio', 'Desercion']
df_modelo = df[variables_modelo]

# %%
df_modelo = pd.get_dummies(df_modelo, columns=['Genero', 'Distrito', 'Provincia', 'Departamento', 'FamiliarResponsable', 'CantParcialesDesaprobados', 'CantFinalesDesaprobados','ExcedeNotasDesaprobadasNoParcNoFin','ExcedeMitadCursosDesaprobados','GrupoEdad','ExcedePagosAtrasados','TipoColegio'], drop_first=True)

# %% [markdown]
# # CREACION DEL MODELO

# %%
#División de los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X = df_modelo.drop(columns=['Desercion'])
y = df_modelo['Desercion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# Diccionario que guarda las métricas
metricas_dict = {
    "Modelo": [],
    "Tipo_Data": [],
    "Exactitud": [],
    "AUC": [],
    "F1-Score": [],
    "Precisión": [],
    "Recall": [],
    #"Verdaderos Negativos": [],
    #"Falsos Positivos": [],
    #"Falsos Negativos": [],
    #Verdaderos Positivos": [],
    "Puntuación": []
}

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix,f1_score, precision_score, recall_score
import xgboost as xgb

# Función para calcular métricas
def calcular_metricas(model, X_test, y_test, nombre_modelo):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    puntuacion = (0.3 * accuracy) + (0.1 * precision) + (0.3 * recall) + (0.2 * f1) + (0.4 * auc)
    
    metricas_dict["Modelo"].append(nombre_modelo)
    metricas_dict["Tipo_Data"].append("Data Original")
    metricas_dict["Exactitud"].append(accuracy)
    metricas_dict["AUC"].append(auc)
    metricas_dict["F1-Score"].append(f1)
    metricas_dict["Precisión"].append(precision)
    metricas_dict["Recall"].append(recall)
    metricas_dict["Puntuación"].append((puntuacion))

# %% [markdown]
# ## MODELOS SIN BALANCEO

# %%
modelos = [
    ("Regresión Logística", LogisticRegression(max_iter=1000)),
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("XGBoost", xgb.XGBClassifier(eval_metric='logloss', random_state=42)),
    ("SVM", SVC(probability=True, random_state=42)),
    ("KNN",KNeighborsClassifier(n_neighbors=5))
]

# %%
# Entrenar y evaluar los modelos
for nombre_modelo, modelo in modelos:
    modelo.fit(X_train, y_train)
    calcular_metricas(modelo, X_test, y_test, nombre_modelo)

# %% [markdown]
# ## MODELOS CON BALANCEO

# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Reentrenar modelos con datos balanceados
for nombre_modelo, modelo in modelos:
    modelo.fit(X_resampled, y_resampled)
    calcular_metricas(modelo, X_test, y_test, f"{nombre_modelo} (SMOTE)")

# %% [markdown]
# ## MODELOS CON BALANCEO Y AJUSTE DE HIPERPARAMETROS

# %%
from sklearn.model_selection import GridSearchCV

# Definir los hiperparámetros para cada modelo
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# %%
# Diccionario con modelos y sus respectivos grids
modelos_param_grids = [
    ("Regresión Logística", LogisticRegression(max_iter=1000), param_grid_lr),
    ("Random Forest", RandomForestClassifier(random_state=42), param_grid_rf),
    ("XGBoost", xgb.XGBClassifier(eval_metric='logloss', random_state=42), param_grid_xgb),
    ("SVM", SVC(probability=True, random_state=42), param_grid_svm),
    ("KNN", KNeighborsClassifier(), param_grid_knn)
]

# %%
# Realizamos la búsqueda de hiperparámetros
for nombre_modelo, modelo, param_grid in modelos_param_grids:
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)
    
    # Entrenamos con los mejores hiperparámetros
    best_model = grid_search.best_estimator_
    print(f"Mejores parámetros para {nombre_modelo}: {grid_search.best_params_}")
    
    # Calculamos métricas con los mejores hiperparámetros
    calcular_metricas(best_model, X_test, y_test, f"{nombre_modelo} (Optimizado y con SMOTE)")

# %% [markdown]
# # EXPORTADO DE METRICAS

# %%
df_metricas = pd.DataFrame(metricas_dict)

df_metricas = df_metricas.sort_values(by="Puntuación", ascending=False)
df_metricas.to_csv(os.path.join(ruta_output,'Modelo_Desercion_Metricas.csv'))
print(df_metricas)

# %%



