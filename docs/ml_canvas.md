# ML Canvas — Previsão de Churn em Telecomunicações

> Documento que traduz o problema de negócio em especificações técnicas,
> seguindo a metodologia CRISP-DM (fase de Business Understanding).

---

## 1. Problema de Negócio

### Contexto

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.
A diretoria precisa identificar quais clientes têm maior risco de cancelamento
(churn) para que ações preventivas de retenção possam ser tomadas antes
que o cancelamento aconteça.

### Objetivo de Negócio

Reduzir a taxa de churn mensal da operadora, permitindo que equipes de
retenção atuem proativamente nos clientes com maior risco de cancelamento.

### Objetivo Técnico (Data Science)

Construir um modelo de classificação binária que, dado o perfil e histórico
de um cliente, retorne a probabilidade de ele cancelar o serviço.

---

## 2. Stakeholders

| Stakeholder | Papel | Interesse |
|---|---|---|
| Diretoria Executiva | Sponsor do projeto | Redução de churn e aumento de receita recorrente |
| Gerência de Retenção | Usuário principal | Receber lista priorizada de clientes em risco para campanhas de retenção |
| Equipe de Marketing | Usuário secundário | Personalizar ofertas de retenção com base no perfil de risco |
| Equipe de Atendimento (SAC) | Usuário operacional | Alertas em tempo real durante interações com clientes em risco |
| Equipe de Dados / ML Engineering | Desenvolvedor | Construir, manter e monitorar o modelo em produção |
| Área Jurídica / Compliance | Regulador interno | Garantir uso ético dos dados e conformidade com LGPD |

---

## 3. Métricas

### Métricas de Negócio (KPIs)

| KPI | Definição | Meta |
|---|---|---|
| Taxa de Churn Mensal | % de clientes que cancelam por mês | Reduzir em 15-20% vs. baseline atual |
| Custo de Churn Evitado | Receita preservada por retenção bem-sucedida | Maximizar (depende do custo médio por cliente) |
| ROI da Campanha de Retenção | (Receita salva - Custo campanha) / Custo campanha | > 3x |
| Taxa de Contato Efetivo | % de clientes contatados que permaneceram | > 40% |

### Métricas Técnicas (Modelo)

| Métrica | Justificativa | Meta |
|---|---|---|
| AUC-ROC | Mede capacidade geral de discriminação entre churn/não-churn | >= 0.80 |
| PR-AUC (Precision-Recall AUC) | Mais informativa em datasets desbalanceados (churn é minoria) | >= 0.60 |
| F1-Score | Equilíbrio entre precision e recall | >= 0.65 |
| Recall | Importante: queremos capturar o máximo de churners possível | >= 0.75 |

**Análise de Trade-off (Falso Positivo vs. Falso Negativo):**

- **Falso Negativo (prediz "fica" mas cancela):** Alto custo — perdemos o cliente sem tentar retê-lo. Custo estimado: receita mensal x tempo médio de vida restante do cliente.
- **Falso Positivo (prediz "cancela" mas fica):** Custo moderado — gastamos recursos de retenção desnecessariamente (desconto, ligação), mas não perdemos o cliente.
- **Decisão:** Priorizamos **recall alto** (capturar mais churners), aceitando mais falsos positivos, pois o custo de perder um cliente é muito maior que o custo de uma campanha de retenção desnecessária.

---

## 4. SLOs (Service Level Objectives)

> SLOs definem os requisitos não-funcionais do modelo em produção.
> Analogia Java: são como os SLAs que você define para uma API REST —
> latência máxima, disponibilidade, throughput.

| SLO | Especificação | Justificativa |
|---|---|---|
| Latência da API | p95 < 200ms por predição | Permite uso em tempo real durante atendimento |
| Disponibilidade | >= 99.5% uptime mensal | Campanhas de retenção não podem parar |
| Throughput | >= 100 predições/segundo | Suporta batch diário de toda a base de clientes |
| Freshness do Modelo | Re-treino mensal ou quando drift > threshold | Comportamento de clientes muda ao longo do tempo |
| Tamanho do Modelo | < 50MB serializado | Viabiliza deploy em containers com recursos limitados |

---

## 5. Dados

### Dataset

**Telco Customer Churn (IBM)** — dataset público com ~7.043 registros e 21 features.

### Variáveis Esperadas

| Categoria | Variáveis | Tipo |
|---|---|---|
| Demográficas | gender, SeniorCitizen, Partner, Dependents | Categóricas |
| Conta | tenure, Contract, PaperlessBilling, PaymentMethod | Mistas |
| Serviços | PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, etc. | Categóricas |
| Financeiras | MonthlyCharges, TotalCharges | Numéricas |
| **Target** | **Churn** (Yes/No) | **Binária** |

### Data Readiness (Checklist)

- [ ] Dados disponíveis e acessíveis (CSV/Kaggle)
- [ ] Volume suficiente (>= 5.000 registros) — OK, ~7.043
- [ ] Features >= 10 — OK, 20 features
- [ ] Variável alvo definida (Churn)
- [ ] Sem restrições legais (dataset público)
- [ ] Qualidade verificada (missing values, tipos, outliers)

---

## 6. Requisitos e Restrições

### Requisitos Funcionais

- Endpoint REST `/predict` que recebe features de um cliente e retorna probabilidade de churn
- Endpoint `/health` para monitoramento
- Validação de entrada com Pydantic
- Logging estruturado de todas as predições

### Requisitos Não-Funcionais

- Reprodutibilidade: seeds fixados, pipeline determinístico
- Testabilidade: >= 3 testes automatizados (smoke, schema, API)
- Manutenibilidade: código modular em src/, linting com ruff
- Rastreabilidade: todos os experimentos registrados no MLflow

### Restrições

| Restrição | Descrição |
|---|---|
| Dados | Dataset público apenas (sem dados proprietários) |
| Modelo obrigatório | MLP (rede neural) treinada com PyTorch |
| Prazo | Entrega conforme calendário da pós-graduação |
| Orçamento | Free tier AWS (limitações de CPU/memória) |
| Equipe | Projeto individual / pequeno grupo |

---

## 7. Abordagem Técnica

### Arquitetura de Deploy: Real-time (API REST)

**Justificativa:** A operadora precisa de predições tanto em batch (rodar toda a
base diariamente para campanhas) quanto em tempo real (durante atendimento
no SAC). Uma API REST atende ambos os cenários — o batch chama a API em
loop, e o SAC chama pontualmente.

### Pipeline

```
Dados brutos → Pré-processamento (sklearn Pipeline) → Feature Engineering
    → Treinamento (PyTorch MLP + baselines sklearn)
    → Avaliação (métricas + MLflow tracking)
    → Serialização do modelo
    → API FastAPI (/predict)
    → Deploy (Docker → AWS ECS Fargate)
```

### Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| API | FastAPI + Uvicorn |
| Validação | Pydantic |
| Modelo | PyTorch (MLP) |
| Pré-processamento | Scikit-Learn Pipeline |
| Tracking | MLflow |
| Testes | pytest |
| Linting | ruff |
| Build/Config | pyproject.toml + Makefile |
| Container | Docker + Compose |
| Cloud | AWS ECS Fargate + ALB |

---

## 8. Riscos e Mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Dataset desbalanceado (churn ~26%) | Modelo enviesado para classe majoritária | Técnicas de balanceamento, métricas adequadas (PR-AUC), class weights |
| Overfitting do MLP | Baixa generalização | Early stopping, validação cruzada, dropout |
| Data drift em produção | Performance degrada com o tempo | Plano de monitoramento, re-treino periódico |
| Falta de experiência em PyTorch | Atraso no desenvolvimento | Baselines simples primeiro, MLP incremental |
| Custo AWS | Exceder free tier | Usar recursos mínimos (256 CPU, 512 MB) |

---

## 9. Critérios de Sucesso

O projeto será considerado bem-sucedido se:

1. O modelo MLP superar os baselines (DummyClassifier, Regressão Logística) em pelo menos 2 das 4 métricas definidas
2. A API responder corretamente a requisições de predição com latência < 200ms
3. Todos os testes automatizados passarem
4. O código passar no linting (ruff) sem erros
5. Os experimentos estiverem rastreados no MLflow com parâmetros, métricas e artefatos
6. A documentação (Model Card + README) estiver completa
