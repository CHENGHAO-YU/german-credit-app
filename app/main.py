import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import shap
import io
import base64

# 导入自定义模块
from data_processor import DataProcessor
from model_trainer import ModelTrainer

# 页面配置
st.set_page_config(
    page_title="German Credit Risk Analysis",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #f5f0e1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'processor' not in st.session_state:
    st.session_state.processor = DataProcessor()
if 'trainer' not in st.session_state:
    st.session_state.trainer = ModelTrainer()

def main():
    # 标题
    st.markdown('<h1 class="main-header">💳 German Credit Risk Analysis System</h1>', 
                unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/bank-card-back-side.png", width=100)
        st.title("🎯 Navigation")
        
        page = st.radio(
            "Select Function:",
            ["📊 Data Upload & EDA", 
             "🤖 Model Training", 
             "📈 Model Evaluation",
             "🔮 Make Predictions",
             "📚 Model Interpretation",
             "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### 📌 Quick Stats")
        if st.session_state.data is not None:
            st.metric("Total Records", len(st.session_state.data))
            st.metric("Features", len(st.session_state.data.columns))
            if 'Risk' in st.session_state.data.columns:
                risk_ratio = (st.session_state.data['Risk'] == 'Risk').mean()
                st.metric("Risk Ratio", f"{risk_ratio:.2%}")
    
    # 主页面内容
    if page == "📊 Data Upload & EDA":
        show_data_upload_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "📈 Model Evaluation":
        show_model_evaluation_page()
    elif page == "🔮 Make Predictions":
        show_prediction_page()
    elif page == "📚 Model Interpretation":
        show_interpretation_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_data_upload_page():
    """数据上传和EDA页面"""
    st.header("📊 Data Upload & Exploratory Data Analysis")
    
    # 数据上传
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file (German Credit Dataset)",
            type=['csv'],
            help="Upload your German Credit training data"
        )
        
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(st.session_state.data)} records!")
    
    with col2:
        st.info("""
        **Expected columns:**
        - CheckingStatus
        - Duration
        - CreditHistory
        - LoanPurpose
        - CreditAmount
        - Risk (target)
        - etc.
        """)
    
    # 使用示例数据按钮
    if st.button("📥 Load Sample Data"):
        st.session_state.data = create_sample_data()
        st.success("✅ Sample data loaded!")
    
    # EDA部分
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # 数据概览
        st.subheader("📋 Data Overview")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Data Sample", "Statistics", "Missing Values", "Distributions"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(df.describe(include='all'), use_container_width=True)
        
        with tab3:
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum(),
                'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        
        with tab4:
            # 风险分布
            if 'Risk' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_counts = df['Risk'].value_counts()
                    fig = px.pie(values=risk_counts.values, 
                               names=risk_counts.index,
                               title="Risk Distribution",
                               color_discrete_map={'Risk': '#FF6B6B', 'No Risk': '#4ECDC4'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 数值列分布
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        selected_col = st.selectbox("Select numeric column:", numeric_cols)
                        fig = px.histogram(df, x=selected_col, color='Risk' if 'Risk' in df.columns else None,
                                         title=f"Distribution of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            # 相关性热图
            st.subheader("🔥 Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, 
                              text_auto=True,
                              color_continuous_scale='RdBu',
                              title="Feature Correlation Matrix")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

def show_model_training_page():
    """模型训练页面"""
    st.header("🤖 Model Training")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.data
    
    # 训练配置
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.3)
    
    with col2:
        n_trials = st.number_input("Optuna Trials", 10, 100, 30)
    
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    # 开始训练按钮
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training model... This may take a few minutes..."):
            # 数据预处理
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Step 1/5: Preprocessing data...")
            progress_bar.progress(20)
            
            processor = st.session_state.processor
            df_processed = processor.preprocess_data(df, is_training=True)
            
            # 分离特征和目标
            X = df_processed.drop(columns=['Risk'])
            y = df_processed['Risk']
            
            # 创建预处理器
            trainer = st.session_state.trainer
            preprocessor = trainer.create_preprocessor(processor.onehot_cols)
            
            # 划分数据集
            status_text.text("Step 2/5: Splitting data...")
            progress_bar.progress(30)
            
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # 转换特征
            X_train_transformed = preprocessor.fit_transform(X_train_raw)
            X_val_transformed = preprocessor.transform(X_val_raw)
            trainer.feature_names = preprocessor.get_feature_names_out()
            
            # 超参数优化
            status_text.text(f"Step 3/5: Optimizing hyperparameters ({n_trials} trials)...")
            progress_bar.progress(50)
            
            study = trainer.optimize_hyperparameters(X_train_transformed, y_train, n_trials)
            
            # 训练模型
            status_text.text("Step 4/5: Training final model...")
            progress_bar.progress(70)
            
            trainer.train_model(X_train_transformed, y_train)
            
            # 应用特征缩放
            X_train_scaled = processor.apply_feature_scaling(
                X_train_transformed, trainer.feature_names, trainer.scaled_importance
            )
            X_val_scaled = processor.apply_feature_scaling(
                X_val_transformed, trainer.feature_names, trainer.scaled_importance
            )
            
            # 重新训练
            trainer.train_model(X_train_scaled, y_train)
            
            # 评估模型
            status_text.text("Step 5/5: Evaluating model...")
            progress_bar.progress(90)
            
            metrics, y_pred, y_pred_proba = trainer.evaluate_model(X_val_scaled, y_val)
            
            # 保存结果到session state
            st.session_state.model_trained = True
            st.session_state.metrics = metrics
            st.session_state.X_val = X_val_scaled
            st.session_state.y_val = y_val
            st.session_state.y_pred = y_pred
            st.session_state.y_pred_proba = y_pred_proba
            
            progress_bar.progress(100)
            status_text.text("✅ Training completed!")
            
            # 显示结果
            st.success("🎉 Model training completed successfully!")
            
            # 显示最佳参数
            st.subheader("🎯 Best Hyperparameters")
            params_df = pd.DataFrame([trainer.best_params])
            st.dataframe(params_df, use_container_width=True)
            
            # 显示性能指标
            st.subheader("📊 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision (Risk)", 
                         f"{metrics['classification_report']['1']['precision']:.4f}")
            with col3:
                st.metric("Recall (Risk)", 
                         f"{metrics['classification_report']['1']['recall']:.4f}")
            with col4:
                st.metric("F1-Score (Risk)", 
                         f"{metrics['classification_report']['1']['f1-score']:.4f}")

def show_model_evaluation_page():
    """模型评估页面"""
    st.header("📈 Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first!")
        return
    
    metrics = st.session_state.metrics
    y_val = st.session_state.y_val
    y_pred = st.session_state.y_pred
    y_pred_proba = st.session_state.y_pred_proba
    
    # 创建评估图表
    col1, col2 = st.columns(2)
    
    with col1:
        # 混淆矩阵
        st.subheader("🎯 Confusion Matrix")
        conf_matrix = metrics['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['No Risk', 'Risk'],
            y=['No Risk', 'Risk'],
            text=conf_matrix,
            texttemplate="%{text}",
            colorscale='Blues'
        ))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC曲线
        st.subheader("📊 ROC Curve")
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.2f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=1, dash='dash')
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Precision-Recall曲线
    st.subheader("📈 Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall curve',
        line=dict(color='green', width=2)
    ))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 特征重要性
    st.subheader("🏆 Feature Importance")
    trainer = st.session_state.trainer
    importance_df = pd.DataFrame({
        'Feature': trainer.feature_names,
        'Importance': trainer.scaled_importance
    }).sort_values('Importance', ascending=False).head(20)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', 
                orientation='h', title="Top 20 Most Important Features")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_page():
    """预测页面"""
    st.header("🔮 Make Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first!")
        return
    
    # 选择输入方式
    input_method = st.radio("Select input method:", ["Upload CSV", "Manual Input"])
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file for prediction", type=['csv'])
        
        if uploaded_file is not None:
            test_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(test_data.head(), use_container_width=True)
            
            if st.button("🔮 Make Predictions"):
                with st.spinner("Making predictions..."):
                    # 预处理数据
                    processor = st.session_state.processor
                    trainer = st.session_state.trainer
                    
                    test_processed = processor.preprocess_data(test_data, is_training=False)
                    
                    # 如果有Id列，保存并删除
                    has_id = 'Id' in test_processed.columns
                    if has_id:
                        ids = test_processed['Id']
                        test_processed = test_processed.drop(columns=['Id'])
                    
                    # 转换特征
                    X_test_transformed = trainer.preprocessor.transform(test_processed)
                    X_test_scaled = processor.apply_feature_scaling(
                        X_test_transformed, trainer.feature_names, trainer.scaled_importance
                    )
                    
                    # 预测
                    predictions = trainer.model.predict(X_test_scaled)
                    predictions_proba = trainer.model.predict_proba(X_test_scaled)
                    
                    # 创建结果DataFrame
                    results = pd.DataFrame({
                        'Prediction': ['Risk' if p == 1 else 'No Risk' for p in predictions],
                        'Risk_Probability': predictions_proba[:, 1],
                        'No_Risk_Probability': predictions_proba[:, 0]
                    })
                    
                    if has_id:
                        results.insert(0, 'Id', ids)
                    
                    st.success("✅ Predictions completed!")
                    st.dataframe(results, use_container_width=True)
                    
                    # 下载按钮
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )
    
    else:  # Manual Input
        st.subheader("Enter customer information:")
        
        # 创建输入表单
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                checking_status = st.selectbox("Checking Status", 
                    ['no_checking', 'less_0', '0_to_200', 'greater_200'])
                duration = st.number_input("Duration (months)", 1, 72, 12)
                credit_history = st.selectbox("Credit History",
                    ['outstanding_credit', 'prior_payments_delayed', 'no_credits', 
                     'credits_paid_to_date', 'all_credits_paid_back'])
            
            with col2:
                loan_purpose = st.selectbox("Loan Purpose",
                    ['car', 'furniture', 'radio/TV', 'domestic appliances', 
                     'repairs', 'education', 'business', 'vacation', 'other'])
                credit_amount = st.number_input("Credit Amount", 250, 20000, 1000)
                savings = st.selectbox("Existing Savings",
                    ['unknown', 'less_100', '100_to_500', '500_to_1000', 'greater_1000'])
            
            with col3:
                employment = st.selectbox("Employment Duration",
                    ['unemployed', 'less_1', '1_to_4', '4_to_7', 'greater_7'])
                sex = st.selectbox("Sex", ['male', 'female'])
                age = st.number_input("Age", 18, 100, 30)
            
            submitted = st.form_submit_button("Predict Risk", type="primary")
            
            if submitted:
                # 创建输入数据DataFrame
                input_data = pd.DataFrame({
                    'CheckingStatus': [checking_status],
                    'Duration': [duration],
                    'CreditHistory': [credit_history],
                    'LoanPurpose': [loan_purpose],
                    'CreditAmount': [credit_amount],
                    'ExistingSavings': [savings],
                    'EmploymentDuration': [employment],
                    'Sex': [sex],
                    'Age': [age],
                    # 添加其他必要的列（使用默认值）
                    'InstallmentRate': [2],
                    'OthersOnLoan': ['none'],
                    'CurrentResidenceDuration': [2],
                    'OwnsProperty': ['real_estate'],
                    'Installments': [1],
                    'InstallmentPlans': ['none'],
                    'Housing': ['own'],
                    'ExistingCreditsCount': [1],
                    'Job': ['skilled'],
                    'Dependents': [1],
                    'Telephone': ['yes'],
                    'ForeignWorker': ['yes']
                })
                
                # 预测
                processor = st.session_state.processor
                trainer = st.session_state.trainer
                
                input_processed = processor.preprocess_data(input_data, is_training=False)
                X_input_transformed = trainer.preprocessor.transform(input_processed)
                X_input_scaled = processor.apply_feature_scaling(
                    X_input_transformed, trainer.feature_names, trainer.scaled_importance
                )
                
                prediction = trainer.model.predict(X_input_scaled)[0]
                prediction_proba = trainer.model.predict_proba(X_input_scaled)[0]
                
                # 显示结果
                st.markdown("---")
                st.subheader("🎯 Prediction Result")
                
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 1:
                        st.error(f"⚠️ **Risk**: High credit risk detected")
                    else:
                        st.success(f"✅ **No Risk**: Low credit risk")
                
                with col2:
                    st.metric("Risk Probability", f"{prediction_proba[1]:.2%}")
                    st.metric("No Risk Probability", f"{prediction_proba[0]:.2%}")

def show_interpretation_page():
    """模型解释页面"""
    st.header("📚 Model Interpretation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first!")
        return
    
    st.info("🔍 SHAP (SHapley Additive exPlanations) helps explain model predictions")
    
    # 获取数据
    X_val = st.session_state.X_val
    trainer = st.session_state.trainer
    
    # 计算SHAP值
    if st.button("Calculate SHAP Values"):
        with st.spinner("Calculating SHAP values... This may take a moment..."):
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(trainer.model)
            
            # 采样数据以加快计算
            sample_size = min(100, len(X_val))
            X_sample = X_val.sample(n=sample_size, random_state=42)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(X_sample)
            
            # SHAP Summary Plot
            st.subheader("📊 SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, 
                            X_sample, plot_type="bar", show=False)
            st.pyplot(fig)
            
            # SHAP Detailed Plot
            st.subheader("🔍 SHAP Detailed Plot")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, 
                            X_sample, show=False)
            st.pyplot(fig2)
            
            st.success("✅ SHAP analysis completed!")

def show_about_page():
    """关于页面"""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 📌 Project Overview
    
    This is a comprehensive **German Credit Risk Analysis System** built with Streamlit,
    designed to predict credit risk using machine learning techniques.
    
    ### 🎯 Key Features
    
    - **Data Upload & EDA**: Upload your data and perform exploratory data analysis
    - **Automated Model Training**: Uses LightGBM with Optuna hyperparameter optimization
    - **Model Evaluation**: Comprehensive evaluation metrics and visualizations
    - **Real-time Predictions**: Make predictions on new data
    - **Model Interpretation**: SHAP values for understanding model decisions
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **ML Framework**: LightGBM
    - **Hyperparameter Tuning**: Optuna
    - **Model Interpretation**: SHAP
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib, Seaborn
    
    ### 📊 Dataset Information
    
    The German Credit dataset contains information about 1000 loan applicants with 20 features:
    - Personal information (age, sex, etc.)
    - Financial status (checking account, savings, etc.)
    - Loan details (amount, duration, purpose)
    - Credit history
    
    ### 🎓 Model Details
    
    - **Algorithm**: LightGBM (Gradient Boosting)
    - **Optimization**: Bayesian optimization via Optuna
    - **Feature Engineering**: Ordinal encoding + One-hot encoding
    - **Feature Scaling**: Importance-based scaling
    
    ### 📈 Performance Metrics
    
    The model is evaluated using:
    - Accuracy
    - Precision & Recall
    - F1-Score
    - ROC-AUC
    - Confusion Matrix
    
    ### 👨‍💻 Author
    
    Developed for the Data Science course assignment.
    
    ### 🔗 Links
    
    - [GitHub Repository](https://github.com/yourusername/german-credit-risk)
    - [Docker Image](https://hub.docker.com/r/yourusername/german-credit-app)
    
    ### 📄 License
    
    MIT License - Feel free to use and modify!
    """)

def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'CheckingStatus': np.random.choice(['no_checking', 'less_0', '0_to_200', 'greater_200'], n_samples),
        'Duration': np.random.randint(6, 72, n_samples),
        'CreditHistory': np.random.choice(['outstanding_credit', 'prior_payments_delayed', 'no_credits', 
                                         'credits_paid_to_date', 'all_credits_paid_back'], n_samples),
        'LoanPurpose': np.random.choice(['car', 'furniture', 'radio/TV', 'domestic appliances', 
                                        'repairs', 'education', 'business', 'vacation', 'other'], n_samples),
        'CreditAmount': np.random.randint(250, 20000, n_samples),
        'ExistingSavings': np.random.choice(['unknown', 'less_100', '100_to_500', 
                                           '500_to_1000', 'greater_1000'], n_samples),
        'EmploymentDuration': np.random.choice(['unemployed', 'less_1', '1_to_4', 
                                              '4_to_7', 'greater_7'], n_samples),
        'InstallmentRate': np.random.randint(1, 5, n_samples),
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'OthersOnLoan': np.random.choice(['none', 'co-applicant', 'guarantor'], n_samples),
        'CurrentResidenceDuration': np.random.randint(1, 5, n_samples),
        'OwnsProperty': np.random.choice(['real_estate', 'savings_insurance', 'car_other', 'unknown'], n_samples),
        'Age': np.random.randint(18, 75, n_samples),
        'InstallmentPlans': np.random.choice(['none', 'bank', 'stores'], n_samples),
        'Housing': np.random.choice(['own', 'rent', 'free'], n_samples),
        'ExistingCreditsCount': np.random.randint(1, 5, n_samples),
        'Job': np.random.choice(['unemployed', 'unskilled', 'skilled', 'highly_skilled'], n_samples),
        'Dependents': np.random.randint(1, 3, n_samples),
        'Telephone': np.random.choice(['none', 'yes'], n_samples),
        'ForeignWorker': np.random.choice(['yes', 'no'], n_samples),
        'Risk': np.random.choice(['Risk', 'No Risk'], n_samples, p=[0.3, 0.7])
    })
    
    return data

if __name__ == "__main__":
    main()