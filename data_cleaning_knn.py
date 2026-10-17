import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer

# Configuración de visualización
plt.style.use('ggplot')

def ejecutar_limpieza_y_analisis():
    # 1. Generación de dataset con ruido sintético
    data = {
        'edad': [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40], # 120 es valor erróneo
        'ingreso_mensual': [15000, 22000, 18000, np.nan, 12000, 950000, 25000, 21000, 19000, np.nan], # 950k es outlier
        'scoring_credito': [650, 700, 680, 710, 590, 800, 670, 640, 620, 730]
    }
    df_dirty = pd.DataFrame(data)

    print("==================================================")
    print("=== 1. DATASET SUCIO ORIGINAL ===")
    print("==================================================")
    print(df_dirty)

    # Visualización inicial (Boxplot antes del tratamiento)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df_dirty['ingreso_mensual'], ax=axes[0], color='salmon')
    axes[0].set_title('Ingreso Mensual (Con Outlier: $950,000)')
    
    sns.boxplot(y=df_dirty['edad'], ax=axes[1], color='lightskyblue')
    axes[1].set_title('Edad (Con Error: 120 años)')
    plt.tight_layout()
    plt.savefig('boxplot_antes_limpieza.png')
    plt.close()
    print("\n[Gráfico guardado: boxplot_antes_limpieza.png]")

    # 2. Tratamiento de Valores Atípicos por Rango Intercuartílico (IQR)
    Q1 = df_dirty['ingreso_mensual'].quantile(0.25)
    Q3 = df_dirty['ingreso_mensual'].quantile(0.75)
    IQR = Q3 - Q1
    limite_superior = Q3 + 1.5 * IQR
    print(f"\n[Análisis IQR] Q1: ${Q1:,.2f} | Q3: ${Q3:,.2f} | IQR: ${IQR:,.2f}")
    print(f"[Análisis IQR] Límite Superior de Ingreso: ${limite_superior:,.2f}")

    # Reemplazo de outliers y errores lógicos por NaN
    df_preprocessed = df_dirty.copy()
    df_preprocessed.loc[df_preprocessed['ingreso_mensual'] > limite_superior, 'ingreso_mensual'] = np.nan
    df_preprocessed.loc[df_preprocessed['edad'] > 100, 'edad'] = np.nan

    print("\n==================================================")
    print("=== 2. DATASET CON OUTLIERS REEMPLAZADOS POR NaN ===")
    print("==================================================")
    print(df_preprocessed)

    # 3. Imputación Múltiple con KNNImputer (k=2)
    imputer_k2 = KNNImputer(n_neighbors=2)
    df_knn_k2 = pd.DataFrame(imputer_k2.fit_transform(df_preprocessed), columns=df_dirty.columns)

    # Experimentos adicionales: KNN (k=3), Media y Mediana
    imputer_k3 = KNNImputer(n_neighbors=3)
    df_knn_k3 = pd.DataFrame(imputer_k3.fit_transform(df_preprocessed), columns=df_dirty.columns)

    mean_imputer = SimpleImputer(strategy='mean')
    df_mean = pd.DataFrame(mean_imputer.fit_transform(df_preprocessed), columns=df_dirty.columns)

    median_imputer = SimpleImputer(strategy='median')
    df_median = pd.DataFrame(median_imputer.fit_transform(df_preprocessed), columns=df_dirty.columns)

    print("\n==================================================")
    print("=== 3. DATASET LIMPIO E IMPUTADO CON KNN (k=2) ===")
    print("==================================================")
    print(df_knn_k2)

    print("\n==================================================")
    print("=== 4. COMPARATIVA DE TÉCNICAS DE IMPUTACIÓN ===")
    print("==================================================")
    comparativa = pd.DataFrame({
        'Original_Preprocesado': df_preprocessed['ingreso_mensual'],
        'KNN_k2': df_knn_k2['ingreso_mensual'],
        'KNN_k3': df_knn_k3['ingreso_mensual'],
        'Media': df_mean['ingreso_mensual'],
        'Mediana': df_median['ingreso_mensual']
    })
    print(comparativa)

    # Visualización final (Boxplot después del tratamiento)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df_knn_k2['ingreso_mensual'], ax=axes[0], color='lightgreen')
    axes[0].set_title('Ingreso Mensual Imputado (KNN k=2)')
    
    sns.boxplot(y=df_knn_k2['edad'], ax=axes[1], color='plum')
    axes[1].set_title('Edad Imputada (KNN k=2)')
    plt.tight_layout()
    plt.savefig('boxplot_despues_limpieza.png')
    plt.close()
    print("\n[Gráfico guardado: boxplot_despues_limpieza.png]")

if __name__ == '__main__':
    ejecutar_limpieza_y_analisis()
