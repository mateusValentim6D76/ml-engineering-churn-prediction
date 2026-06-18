# Model Card - Churn Prediction MLP

**Versao do modelo:** 0.1.0
**Data de criacao:** Junho 2025
**Autor:** Mateus Valentim
**Contexto academico:** Tech Challenge - Pos-Graduacao em ML Engineering (FIAP)

---

## 1. Descricao do Modelo

Rede Neural Multilayer Perceptron (MLP) treinada com PyTorch para prever a probabilidade de cancelamento (churn) de clientes de uma operadora de telecomunicacoes. O modelo recebe atributos demograficos, contratuais e de uso do cliente e retorna uma probabilidade entre 0 e 1.

---

## 2. Uso Pretendido

### Casos de uso previstos

- Identificacao antecipada de clientes com risco de cancelamento
- Priorizacao de acoes de retencao (ofertas, contato proativo)
- Segmentacao de base de clientes por risco

### Casos de uso fora do escopo

- Tomada de decisao automatica sem revisao humana
- Aplicacao em segmentos de mercado diferentes de telecomunicacoes
- Uso em contextos regulados sem auditoria adicional (ex.: credito, seguros)

### Usuarios esperados

Equipes de CRM, marketing e retencao de clientes de operadoras de telecomunicacoes.

---

## 3. Dados de Treinamento

| Item | Detalhe |
|------|---------|
| Dataset | Telco Customer Churn (IBM) |
| Fonte | [Kaggle - blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Versao | v1.0-telco-ibm |
| Registros | 7.043 clientes |
| Features originais | 19 (demograficas, servicos contratados, financeiras) |
| Features apos OHE | 30 |
| Target | `Churn` binario (Yes/No) |
| Desbalanceamento | 26,5% positivos (churn) / 73,5% negativos |
| Divisao | 80% treino / 20% teste (stratified, random_state=42) |

### Distribuicao do target

```
Churn = No  (nao cancela):  73,5%
Churn = Yes (cancela):      26,5%
```

O desbalanceamento foi tratado com `pos_weight` no `BCEWithLogitsLoss`, calculado como `n_negativos / n_positivos`.

### Pre-processamento

1. Remocao de `customerID` (identificador sem valor preditivo)
2. Conversao de `TotalCharges` para numerico (valores em branco tratados como 0)
3. One-Hot Encoding para todas as variaveis categoricas
4. Normalizacao com `StandardScaler` ajustado **somente no conjunto de treino** para evitar data leakage

---

## 4. Arquitetura do Modelo

| Componente | Configuracao |
|-----------|-------------|
| Tipo | MLP (Multi-Layer Perceptron) |
| Framework | PyTorch |
| Camadas ocultas | [256, 128, 64] neuronios |
| Ativacao | ReLU |
| Regularizacao | Dropout (p=0.3) + BatchNorm |
| Saida | Logit unico (sigmoid aplicado na inferencia) |
| Input dim | 30 features |
| Funcao de perda | BCEWithLogitsLoss com pos_weight |
| Otimizador | Adam (lr=1e-3) |
| Early Stopping | patience=15 epocas |
| Max epocas | 200 |
| Batch size | 64 |
| Random seed | 42 |

**Diagrama simplificado:**

```
Input (30)
    |
Linear(30 -> 256) -> BatchNorm -> ReLU -> Dropout(0.3)
    |
Linear(256 -> 128) -> BatchNorm -> ReLU -> Dropout(0.3)
    |
Linear(128 -> 64) -> BatchNorm -> ReLU -> Dropout(0.3)
    |
Linear(64 -> 1) -> Logit
    |
Sigmoid -> Probabilidade de churn [0, 1]
```

---

## 5. Metricas de Desempenho

Avaliadas no conjunto de teste (20% dos dados, 1.409 registros):

| Metrica | Valor |
|---------|-------|
| Accuracy | 0.7324 |
| Precision | 0.4976 |
| Recall | 0.8182 |
| F1-Score | 0.6188 |
| ROC-AUC | **0.8391** |
| PR-AUC | 0.6282 |

### Comparacao com baselines

| Modelo | Accuracy | F1-Score | ROC-AUC |
|--------|----------|----------|---------|
| DummyClassifier (random) | 0.50 | 0.36 | 0.50 |
| Logistic Regression | 0.80 | 0.57 | 0.84 |
| **MLP PyTorch (este modelo)** | **0.73** | **0.62** | **0.84** |

### Interpretacao

O modelo foi otimizado para **alto recall (0.82)**, priorizando a deteccao de clientes que vao cancelar (falsos negativos custam mais que falsos positivos no contexto de retencao). O ROC-AUC de 0.84 indica boa capacidade discriminativa.

O threshold padrao e **0.5**, mas pode ser ajustado conforme o custo do negocio:
- Threshold mais baixo (ex: 0.3) - maior recall, mais alertas falsos
- Threshold mais alto (ex: 0.7) - maior precision, menos alertas

---

## 6. Features do Modelo

As 30 features utilizadas apos One-Hot Encoding:

| # | Feature | Tipo |
|---|---------|------|
| 1 | SeniorCitizen | Numerica |
| 2 | tenure | Numerica |
| 3 | MonthlyCharges | Numerica |
| 4 | TotalCharges | Numerica |
| 5 | gender_Male | Binaria (OHE) |
| 6 | Partner_Yes | Binaria (OHE) |
| 7 | Dependents_Yes | Binaria (OHE) |
| 8 | PhoneService_Yes | Binaria (OHE) |
| 9 | MultipleLines_No phone service | Binaria (OHE) |
| 10 | MultipleLines_Yes | Binaria (OHE) |
| 11 | InternetService_Fiber optic | Binaria (OHE) |
| 12 | InternetService_No | Binaria (OHE) |
| 13 | OnlineSecurity_No internet service | Binaria (OHE) |
| 14 | OnlineSecurity_Yes | Binaria (OHE) |
| 15 | OnlineBackup_No internet service | Binaria (OHE) |
| 16 | OnlineBackup_Yes | Binaria (OHE) |
| 17 | DeviceProtection_No internet service | Binaria (OHE) |
| 18 | DeviceProtection_Yes | Binaria (OHE) |
| 19 | TechSupport_No internet service | Binaria (OHE) |
| 20 | TechSupport_Yes | Binaria (OHE) |
| 21 | StreamingTV_No internet service | Binaria (OHE) |
| 22 | StreamingTV_Yes | Binaria (OHE) |
| 23 | StreamingMovies_No internet service | Binaria (OHE) |
| 24 | StreamingMovies_Yes | Binaria (OHE) |
| 25 | Contract_One year | Binaria (OHE) |
| 26 | Contract_Two year | Binaria (OHE) |
| 27 | PaperlessBilling_Yes | Binaria (OHE) |
| 28 | PaymentMethod_Credit card (automatic) | Binaria (OHE) |
| 29 | PaymentMethod_Electronic check | Binaria (OHE) |
| 30 | PaymentMethod_Mailed check | Binaria (OHE) |

---

## 7. Limitacoes e Riscos

| Limitacao | Descricao |
|-----------|-----------|
| Distribuicao do dataset | Treinado em dataset publico IBM com 7k clientes. Pode nao generalizar para outras operadoras ou regioes |
| Deriva temporal | O modelo nao tem mecanismo de deteccao de data drift. Requer re-treinamento periodico |
| Features estaticas | Nao captura comportamento temporal do cliente (ex.: variacao de uso ao longo do tempo) |
| Interpretabilidade | Modelo de caixa-preta. Nao explica o motivo da predicao sem tecnicas como SHAP/LIME |
| Desbalanceamento | Alta taxa de falsos positivos e inerente ao alto recall. Requer validacao de negocio |

---

## 8. Consideracoes Eticas

- O modelo **nao utiliza** atributos sensiveis como raca, etnia ou renda diretamente
- `gender` e uma das features - avaliar se sua inclusao gera disparidade de tratamento entre grupos
- Decisoes de retencao baseadas no modelo **devem ter revisao humana** para evitar discriminacao ou tratamento indevido
- Os dados de treinamento sao publicos e anonimizados (sem PII)

---

## 9. Reproducibilidade

```powershell
# Clonar o repositorio
git clone https://github.com/mateusValentim6D76/ml-engineering-churn-prediction
cd ml-engineering-churn-prediction

# Instalar dependencias
pip install -e ".[dev]"

# Treinar o modelo (seed fixo: 42)
python -m src.models.train --epochs 200
```

Os artefatos gerados sao salvos em `models/`:
- `churn_mlp.pt` - checkpoint completo com pesos, arquitetura e metricas
- `scaler.joblib` - StandardScaler serializado

---

## 10. Rastreamento de Experimentos

O MLflow registra automaticamente cada execucao de treinamento:

```powershell
# Subir a UI do MLflow
mlflow ui --port 5000
# Acessar: http://localhost:5000
```

Metricas registradas por execucao: accuracy, precision, recall, f1, roc_auc, pr_auc, epochs treinados e tempo de treinamento.

---

## 11. Inferencia via API

```
POST /predict
POST /predict-batch
```

Consulte o [README.md](README.md) para exemplos completos de requisicao e instrucoes de deploy local e na AWS.

---

## 12. Versionamento e Contato

| Campo | Valor |
|-------|-------|
| Versao do modelo | 0.1.0 |
| Versao do dataset | v1.0-telco-ibm |
| Repositorio | https://github.com/mateusValentim6D76/ml-engineering-churn-prediction |
| Contato | mvalentimcloud@gmail.com |
| Licenca | MIT |
