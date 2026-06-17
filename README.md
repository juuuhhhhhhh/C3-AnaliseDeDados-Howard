# 🏠 House Prices — Data Analysis & Machine Learning Hackathon

**Avaliação C3 | FAESA Centro Universitário | Prof. M.Sc. Howard Roatti**

---

## 📌 Objetivo

Explorar um conjunto de dados de preços de casas nos Estados Unidos, aplicando técnicas de análise exploratória, feature engineering, aprendizagem supervisionada e não supervisionada para prever e compreender os fatores que influenciam os preços dos imóveis.

---

## 📁 Estrutura do Repositório

```
house_prices_project/
├── README.md
├── house_prices_analysis.ipynb   ← Notebook principal (análise completa)
├── data/
│   └── train.csv                 ← Dataset (1460 amostras, 27 features)
└── plots/                        ← Visualizações geradas (19 gráficos)
    ├── 01_saleprice_distribution.png
    ├── 02_correlation_saleprice.png
    ├── 03_heatmap_correlation.png
    ├── 04_scatter_top_features.png
    ├── 05_categorical_boxplots.png
    ├── 06_new_features_distribution.png
    ├── 07_regression_metrics.png
    ├── 08_best_regressor_analysis.png
    ├── 09_feature_importance_rf.png
    ├── 10_classification_roc_cm.png
    ├── 11_classification_comparison.png
    ├── 12_kmeans_elbow.png
    ├── 13_kmeans_clusters.png
    ├── 14_cluster_profile_heatmap.png
    ├── 15_pca_analysis.png
    ├── 16_tsne_visualization.png
    ├── 17_apriori_rules.png
    ├── 18_lof_outliers.png
    └── 19_dashboard_final.png
```

---

## 🗂️ Dataset

- **Fonte:** Kaggle — [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Amostras:** 1.460 imóveis
- **Features:** 27 variáveis (numéricas e categóricas)
- **Target:** `SalePrice` — preço de venda do imóvel (USD)

---

## 📋 Etapas do Projeto

### 1. Análise Exploratória de Dados (EDA)
- Identificação de tipos de variáveis (numéricas e categóricas)
- Análise de valores faltantes e outliers
- Distribuição do target (`SalePrice`)
- Matriz de correlação e heatmap
- Scatter plots das features mais correlacionadas
- Boxplots por variáveis categóricas

### 2. Feature Engineering
- Imputação de valores faltantes pela mediana
- Criação de novas features: `HouseAge`, `RemodAge`, `TotalSF`, `TotalBath`, `QualArea`
- Variáveis binárias: `HasPool`, `HasFireplace`, `HasGarage`
- Transformação logarítmica do target
- Label Encoding das variáveis categóricas

### 3. Aprendizagem Supervisionada — Regressão
Modelos treinados e avaliados:
- Regressão Linear, Ridge, Lasso
- KNN Regressor
- Árvore de Decisão
- **Random Forest** ✅ (melhor desempenho)
- Gradient Boosting
- XGBoost

Métricas: `MAE`, `RMSE`, `R²`, `Cross-Validation RMSE`

### 4. Aprendizagem Supervisionada — Classificação
Variável alvo binária: preço alto vs baixo (threshold = mediana)

Modelos treinados:
- Regressão Logística
- Naive Bayes
- KNN Classifier
- Árvore de Decisão
- **Random Forest / Gradient Boosting** ✅ (melhor desempenho)
- XGBoost

Métricas: `Acurácia`, `AUC-ROC`, `Precision`, `Recall`, `F1-Score`

### 5. Aprendizagem Não Supervisionada
| Técnica | Algoritmo | Resultado |
|---|---|---|
| Clusterização | KMeans (k=4) | 4 perfis de imóveis identificados |
| Redução de Dim. | PCA + t-SNE | 4 componentes para 90% da variância |
| Análise de Assoc. | Apriori | Regras: Qual_Alta + Area_Grande → Preço_Alto |
| Detecção de Outlier | LOF | ~5% de outliers detectados |

---

## 📊 Principais Resultados

| Tarefa | Melhor Modelo | Métrica Principal |
|---|---|---|
| Regressão | Random Forest / Gradient Boosting | R² > 0.90 |
| Classificação | Gradient Boosting / XGBoost | AUC-ROC > 0.95 |
| Clusterização | KMeans (k=4) | 4 clusters distintos |
| Outliers | LOF | ~73 outliers (5%) |

---

## 🚀 Como Executar

```bash
# Instalar dependências
pip install numpy pandas matplotlib seaborn scikit-learn xgboost mlxtend

# Executar o notebook
jupyter notebook house_prices_analysis.ipynb
```

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**
- **Pandas / NumPy** — Manipulação de dados
- **Matplotlib / Seaborn** — Visualização
- **Scikit-learn** — ML supervisionado e não supervisionado
- **XGBoost** — Gradient boosting otimizado
- **MLxtend** — Algoritmo Apriori

---

## 📚 Referências

- [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [MLxtend Documentation](http://rasbt.github.io/mlxtend/)
