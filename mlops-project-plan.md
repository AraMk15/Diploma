# MLOps համակարգի մշակման պլան
## Ամպային միջավայրում ML մոդելների ուսուցման, տեղակայման և մոնիթորինգի ավտոմատացում

### Պրոյեկտի նպատակ
Ստեղծել ավտոմատացված MLOps համակարգ, որը կապահովի:
- ML մոդելների ավտոմատ ուսուցում
- Cloud deployment (AWS/GCP/Azure)
- Real-time մոնիթորինգ և alerting
- Model versioning և CI/CD

---

## 1. Ճարտարապետություն

### 1.1 Տեխնոլոգիական stack
- **ML Framework**: PyTorch / TensorFlow / Scikit-learn
- **Orchestration**: Kubeflow / MLflow / Airflow
- **Container**: Docker + Kubernetes
- **Cloud Provider**: AWS (SageMaker) / GCP (Vertex AI) / Azure ML
- **Monitoring**: Prometheus + Grafana
- **CI/CD**: GitHub Actions / GitLab CI
- **Model Registry**: MLflow Model Registry

### 1.2 Համակարգի բաղադրիչներ
```
┌─────────────────────────────────────────────────┐
│         Data Pipeline (Airflow/Kubeflow)        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Extract  │→ │Transform │→ │  Load    │       │
│  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Training Pipeline                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Preprocess│→ │  Train   │→ │ Validate │       │
│  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Model Registry (MLflow)                 │
│  • Version control                              │
│  • Metadata storage                             │
│  • Model artifacts                              │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Deployment (Kubernetes)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Canary  │→ │  Staging │→ │Production│       │
│  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Monitoring (Prometheus/Grafana)         │
│  • Model performance metrics                    │
│  • Data drift detection                         │
│  • System health checks                         │
│  • Alerting                                     │
└─────────────────────────────────────────────────┘
```

---

## 2. ECC Commands - Ինչպես օգտագործել

### Պրոյեկտի սկիզբ
```bash
/plan-prd               # PRD (Product Requirements Document) ստեղծում
/plan                   # Մանրամասն implementation plan
/feature-dev            # Ուղղորդված feature development
```

### Python ML կոդի մշակում
```bash
/python-review          # Python կոդի review և optimization
/python-test            # TDD workflow Python-ի համար
/test-coverage          # Test coverage վերլուծություն և բացակայող tests-ի գեներացում
```

### Infrastructure և DevOps
```bash
/build-fix             # Build error-ների ավտոմատ ուղղում
/code-review           # Ընդհանուր quality check
/security-scan         # Security vulnerabilities-ի ստուգում
```

### Orchestrated Workflows
```bash
/orch-add-feature      # Նոր feature end-to-end (plan → TDD → review → commit)
/orch-build-mvp        # MVP bootstrap-ից մինչև working prototype
/orch-fix-defect       # Bug fix regression test-ի հետ
```

### Multi-Model Collaboration
```bash
/multi-plan            # Բազմամոդել collaborative planning
/multi-backend         # Backend-focused development
```

---

## 3. Իմպլեմենտացիայի քայլեր

### Փուլ 1: Infrastructure Setup (1-2 շաբաթ)
- [ ] Cloud account setup (AWS/GCP/Azure)
- [ ] Kubernetes cluster ստեղծում
- [ ] Docker registry կարգավորում
- [ ] CI/CD pipeline setup
- [ ] **ECC command**: `/plan` infrastructure-ի համար

### Փուլ 2: Data Pipeline (1-2 շաբաթ)
- [ ] Data ingestion system
- [ ] Data validation և cleaning
- [ ] Feature engineering pipeline
- [ ] Data versioning (DVC)
- [ ] **ECC command**: `/python-review` data pipeline կոդի համար

### Փուլ 3: Training Pipeline (2-3 շաբաթ)
- [ ] Training orchestration (Kubeflow/MLflow)
- [ ] Hyperparameter tuning
- [ ] Experiment tracking
- [ ] Model evaluation metrics
- [ ] **ECC command**: `/python-test` training կոդի համար

### Փուլ 4: Model Registry & Versioning (1 շաբաթ)
- [ ] MLflow Model Registry integration
- [ ] Model metadata storage
- [ ] Version control system
- [ ] Model promotion workflow (staging → production)

### Փուլ 5: Deployment System (2 շաբաթ)
- [ ] Model serving API (FastAPI/Flask)
- [ ] Kubernetes deployment manifests
- [ ] Canary deployment strategy
- [ ] Load balancing և scaling
- [ ] **ECC command**: `/build-fix` deployment issues-ի համար

### Փուլ 6: Monitoring & Observability (1-2 շաբաթ)
- [ ] Prometheus metrics collection
- [ ] Grafana dashboards
- [ ] Data drift detection
- [ ] Alerting system (Slack/Email/PagerDuty)
- [ ] Model performance tracking

### Փուլ 7: Testing & Documentation (1 շաբաթ)
- [ ] Unit tests (pytest)
- [ ] Integration tests
- [ ] Load testing (Locust)
- [ ] API documentation (OpenAPI/Swagger)
- [ ] **ECC command**: `/test-coverage` թեստերի համար

---

## 4. Կոդի օրինակներ

### 4.1 Training Pipeline (Python)
```python
# src/training/pipeline.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """ML model training pipeline with MLflow tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        mlflow.set_tracking_uri(config['mlflow_uri'])
        mlflow.set_experiment(config['experiment_name'])
    
    def train(self, data) -> Any:
        """
        Train ML model with experiment tracking.
        
        Args:
            data: Training dataset with features and target
            
        Returns:
            Trained model instance
        """
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config['model_params'])
            
            # Data preparation
            X_train, X_test, y_train, y_test = train_test_split(
                data.features, 
                data.target,
                test_size=self.config['test_size'],
                random_state=42
            )
            
            # Model training
            model = self.config['model_class'](**self.config['model_params'])
            model.fit(X_train, y_train)
            
            # Evaluation
            metrics = self.evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            
            # Save model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=self.config['model_name']
            )
            
            logger.info(f"Training completed. Metrics: {metrics}")
            
        return model
    
    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        """Calculate and return evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
```

### 4.2 Model Serving API (FastAPI)
```python
# src/serving/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import logging
from typing import List, Dict
import prometheus_client
from prometheus_client import Counter, Histogram

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

app = FastAPI(title="ML Model Serving API", version="1.0.0")
logger = logging.getLogger(__name__)

# Load model
model = None

@app.on_event("startup")
async def startup_event():
    """Load ML model on startup."""
    global model
    model_uri = "models:/production/latest"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Model loaded from {model_uri}")

class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[float]

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: List[float]
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
@prediction_latency.time()
async def predict(request: PredictionRequest):
    """
    Make prediction using loaded ML model.
    
    Args:
        request: Features for prediction
        
    Returns:
        Model prediction and version
    """
    try:
        prediction = model.predict([request.features])
        prediction_counter.inc()
        
        return PredictionResponse(
            prediction=prediction.tolist(),
            model_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return prometheus_client.generate_latest()
```

### 4.3 Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-serving
  namespace: mlops
  labels:
    app: ml-model
    version: v1.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
        version: v1.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: model-server
        image: your-registry.io/ml-model:v1.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
  namespace: mlops
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 4.4 CI/CD Pipeline (GitHub Actions)
```yaml
# .github/workflows/mlops-pipeline.yml
name: MLOps CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  train:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Train model
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
      run: |
        python src/training/train.py

  deploy:
    needs: train
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/deployment.yaml
        kubectl rollout status deployment/ml-model-serving -n mlops
```

---

## 5. Մոնիթորինգ և Observability

### 5.1 Prometheus Metrics
```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Model performance metrics
model_predictions_total = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_version', 'status']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Time spent processing prediction',
    buckets=[.001, .005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5]
)

model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_version']
)

data_drift_score = Gauge(
    'data_drift_score',
    'Data drift detection score',
    ['feature_name']
)
```

### 5.2 Grafana Dashboard
```json
{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Predictions per Minute",
        "targets": [
          {
            "expr": "rate(model_predictions_total[1m])"
          }
        ]
      },
      {
        "title": "Prediction Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "targets": [
          {
            "expr": "model_accuracy"
          }
        ]
      }
    ]
  }
}
```

---

## 6. ECC Skills - Օգտագործման օրինակներ

### ML Workflow
```bash
# Կարդալ ML patterns-ը
/skill ml-adoption-playbook

# Python best practices
/skill python-patterns

# Testing strategies
/skill python-testing
```

### Infrastructure
```bash
# Database patterns
/skill postgres-patterns
/skill redis-patterns

# API design
/skill api-design
/skill fastapi-patterns
```

---

## 7. Project Structure
```
mlops-system/
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── monitoring/
│       ├── prometheus.yaml
│       └── grafana.yaml
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── train.py
│   ├── serving/
│   │   ├── __init__.py
│   │   └── api.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── data/
│       ├── __init__.py
│       └── preprocessing.py
├── tests/
│   ├── test_training.py
│   ├── test_api.py
│   └── test_monitoring.py
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── requirements.txt
├── setup.py
└── README.md
```

---

## 8. Ժամանակացույց

| Փուլ | Տևողություն | Կարգավիճակ | ECC Command |
|------|------------|------------|-------------|
| Infrastructure Setup | 1-2 շաբաթ | 🔵 Պլանավորված | `/plan` |
| Data Pipeline | 1-2 շաբաթ | 🔵 Պլանավորված | `/python-review` |
| Training Pipeline | 2-3 շաբաթ | 🔵 Պլանավորված | `/python-test` |
| Model Registry | 1 շաբաթ | 🔵 Պլանավորված | `/feature-dev` |
| Deployment | 2 շաբաթ | 🔵 Պլանավորված | `/build-fix` |
| Monitoring | 1-2 շաբաթ | 🔵 Պլանավորված | `/orch-add-feature` |
| Testing & Docs | 1 շաբաթ | 🔵 Պլանավորված | `/test-coverage` |

**Ընդամենը:** 9-13 շաբաթ

---

## 9. Հաջորդ քայլերը

### Անմիջական (այսօր):
1. Ընտրել cloud provider (AWS/GCP/Azure)
2. Ստեղծել GitHub repository
3. Սահմանել ML use case:
   - Image classification
   - Fraud detection
   - Customer churn prediction
   - Demand forecasting

### Այս շաբաթ:
```bash
# ECC օգտագործելով սկսել
cd ~/projects/mlops-system
git init

# Ստեղծել PRD
/plan-prd

# Մանրամասն պլան
/plan
```

### Առաջիկա 2 շաբաթ:
1. Infrastructure setup
2. Simple ML pipeline prototype
3. CI/CD հիմքեր
4. Basic monitoring

---

## 10. Օգտակար ռեսուրսներ

### Documentation
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Kubernetes](https://kubernetes.io/docs/)
- [Prometheus](https://prometheus.io/docs/)

### Cloud Providers
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [GCP Vertex AI](https://cloud.google.com/vertex-ai)
- [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/)

### Tools
- [DVC (Data Version Control)](https://dvc.org/)
- [Evidently (Model Monitoring)](https://www.evidentlyai.com/)
- [Great Expectations (Data Quality)](https://greatexpectations.io/)

---

## 11. Success Metrics

### Տեխնիկական
- ✅ Model training time < 2 hours
- ✅ Deployment time < 10 minutes
- ✅ API latency p95 < 100ms
- ✅ Test coverage > 80%
- ✅ Uptime > 99.9%

### Բիզնես
- ✅ Model accuracy > 90%
- ✅ Automated retraining weekly
- ✅ Zero-downtime deployments
- ✅ Data drift detection < 24h

---

**Հաջողություն ձեր դիպլոմայինի աշխատանքում! 🚀🤖**

*Ստեղծված ECC-ով - 2026-10-05*
