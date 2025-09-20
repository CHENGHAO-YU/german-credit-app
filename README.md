# 💳 German Credit Risk Analysis System

[![CI/CD Pipeline](https://github.com/CHENGHAO-YU/german-credit-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/CHENGHAO-YU/german-credit-risk/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/docker/pulls/CHENGHAO-YU/german-credit-app.svg)](https://hub.docker.com/r/CHENGHAO-YU/german-credit-app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive credit risk analysis system built with Streamlit and LightGBM for the German Credit dataset.

## 🌟 Features

- 📊 **Interactive Data Exploration**: Upload and analyze credit data
- 🤖 **Automated ML Pipeline**: LightGBM with Optuna hyperparameter optimization  
- 📈 **Comprehensive Evaluation**: Multiple metrics and visualizations
- 🔮 **Real-time Predictions**: Make predictions on new data
- 📚 **Model Interpretation**: SHAP values for explainability

## 🚀 Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/CHENGHAO-YU/german-credit-risk.git
cd german-credit-risk
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app/main.py
```
### Docker
Pull and run the Docker image:
```bash
docker pull CHENGHAO-YU/german-credit-app:latest
docker run -p 8501:8501 CHENGHAO-YU/german-credit-app:latest
```

### Testing
Run tests with coverage:
```bash
pytest tests/ -v --cov=app --cov-report=html
```

### Project Structure
```basic
german-credit-risk/
├── app/                    # Application source code
│   ├── main.py            # Streamlit main application
│   ├── data_processor.py  # Data processing utilities
│   └── model_trainer.py   # Model training and evaluation
├── tests/                  # Test files
├── data/                   # Data directory
├── models/                 # Saved models
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
└── .github/workflows/     # CI/CD pipelines
```

### Technology Stack
Frontend: Streamlit
ML Framework: LightGBM
Hyperparameter Tuning: Optuna
Model Interpretation: SHAP
Data Processing: Pandas, NumPy, Scikit-learn
Visualization: Plotly, Matplotlib, Seaborn
Containerization: Docker
CI/CD: GitHub Actions

### Model Performance
The model achieves the following performance on the validation set:
Accuracy: ~75%
Precision: ~72%
Recall: ~78%
AUC-ROC: ~0.80

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.