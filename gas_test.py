import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
from imblearn.under_sampling import RandomUnderSampler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
warnings.filterwarnings('ignore')
st.set_page_config(page_title="MalGAN Dashboard", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Data Exploration", "Model Training", 
                                "Black Box Model", "GAN Components", "Evaluation", "Adversarial Attack"])

# Load data and models (cached to avoid reloading)
@st.cache_data
def load_data():
    try:
        data_path = 'balanced_dataset.csv'
        dataframe = pd.read_csv(data_path)
        if 'Name' in dataframe.columns:
            dataframe.drop(['Name'], axis=1, inplace=True)
        
        # Basic data validation
        if 'Malware' not in dataframe.columns:
            st.error("Target column 'Malware' not found in dataset")
            return None
            
        # Verify balance
        counts = dataframe['Malware'].value_counts()
        if abs(counts[0] - counts[1]) > 100:  # Allow small imbalance
            st.warning(f"Dataset not perfectly balanced: {counts[0]} benign vs {counts[1]} malware")
            
        return dataframe
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_models():
    models = {
        'blackBox': None,
        'generator': None,
        'substituteDetector': None,
        'scaler': MinMaxScaler()
    }
    
    try:
        with open('blackBox_file.pkl', 'rb') as f:
            models['blackBox'] = pickle.load(f)
        st.sidebar.success("✅ Black Box model loaded")
    except Exception as e:
        st.sidebar.warning(f"Could not load Black Box model: {str(e)}")
    
    try:
        with open('generator.pkl', 'rb') as f:
            models['generator'] = pickle.load(f)
        st.sidebar.success("✅ Generator loaded")
    except Exception as e:
        st.sidebar.warning(f"Could not load Generator: {str(e)}")
    
    try:
        with open('substituteDetector.pkl', 'rb') as f:
            models['substituteDetector'] = pickle.load(f)
        st.sidebar.success("✅ Substitute Detector loaded")
    except Exception as e:
        st.sidebar.warning(f"Could not load Substitute Detector: {str(e)}")
    
    try:
        with open('scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        st.sidebar.success("✅ Scaler loaded")
    except:
        st.sidebar.warning("Could not load scaler - using new instance")
    
    return models

dataframe = load_data()
models = load_models()

def validate_model(model, model_name=""):
    if model is None:
        st.error(f"{model_name} model not loaded")
        return False
    
    try:
        if hasattr(model, 'predict'):
            return True
        return False
    except:
        return False

def get_test_data():
    """Helper function to get consistent test data across pages"""
    if dataframe is None:
        return None, None, None, None
    
    X = dataframe.drop('Malware', axis=1)
    y = dataframe['Malware']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Page 1: Overview
if page == "Overview":
    st.title("Malware Detection Adversarial Network")
    st.markdown("""
    <div class="info-box">
    <h3>Overview</h3>
    <p>This application demonstrates a system to generate adversarial malware samples 
    that can bypass black-box malware detectors while preserving malicious functionality.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Key Components:**
    - **Black Box Detector**: Random Forest classifier trained on balanced malware/benign samples
    - **Generator Network**: Neural network that generates synthetic malware samples
    - **Substitute Detector**: Neural network that approximates the black box detector
    
    **Navigation Guide:**
    1. Explore the balanced dataset
    2. Train all required models
    3. Analyze the Black Box detector
    4. Examine GAN components
    5. Evaluate system performance
    6. Generate adversarial examples
    """)
    
    st.image("https://raw.githubusercontent.com/yanminglai/Malware-GAN/refs/heads/master/figures/Architecture.png", 
             caption="System Architecture", width=600)
    
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p>This tool is for research and educational purposes only. Malicious use of this 
    technology is prohibited. Always obtain proper authorization before testing any 
    security systems.</p>
    </div>
    """, unsafe_allow_html=True)

# Page 2: Data Exploration
elif page == "Data Exploration":
    st.title("Data Exploration")
    
    if dataframe is None:
        st.error("No data loaded. Please check your dataset.")
        st.stop()
    
    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {dataframe.shape}")
    
    with st.expander("View Data Sample"):
        st.dataframe(dataframe.head().style.format("{:.4f}"))
    
    with st.expander("Data Statistics"):
        st.dataframe(dataframe.describe().style.format("{:.4f}"))
    
    st.subheader("Class Distribution Analysis")
    counts = dataframe['Malware'].value_counts()
    ratio = counts[1] / counts[0]
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=counts, names=['Benign', 'Malware'], 
                    title=f"Class Distribution (1:{ratio:.1f} ratio)",
                    color=['Benign', 'Malware'],
                    color_discrete_map={'Benign':'green', 'Malware':'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(x=counts.index.map({0:'Benign', 1:'Malware'}), y=counts.values,
                    title="Sample Counts",
                    labels={'x':'Class', 'y':'Count'},
                    color=counts.index.map({0:'Benign', 1:'Malware'}),
                    color_discrete_map={'Benign':'green', 'Malware':'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    if abs(counts[0] - counts[1]) > 100:
        st.warning(f"Dataset imbalance detected: {counts[0]} benign vs {counts[1]} malware samples")
    else:
        st.success("Dataset is well balanced")
    
    st.subheader("Feature Analysis")
    tab1, tab2 = st.tabs(["Correlation", "Distributions"])
    
    with tab1:
        st.write("Feature Correlation Matrix")
        corr_matrix = dataframe.corr()
        fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', 
                       zmin=-1, zmax=1, aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        selected_feature = st.selectbox("Select feature", dataframe.columns[:-1])
        fig = px.histogram(dataframe, x=selected_feature, color='Malware',
                          nbins=50, marginal='box',
                          color_discrete_map={0:'green', 1:'red'},
                          title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

# Page 3: Model Training
elif page == "Model Training":
    st.title("Model Training Center")
    
    if dataframe is None:
        st.error("No data loaded. Cannot train models.")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <h4>Training with Balanced Data</h4>
    <p>Your dataset contains 5,012 benign and 5,012 malware samples - no imbalance handling required.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Black Box Detector")
        st.markdown("""
        <p>Random Forest classifier serving as the target malware detector.</p>
        """, unsafe_allow_html=True)
        
        with st.form("rf_config"):
            n_estimators = st.slider("Number of trees", 50, 500, 100, 50)
            max_depth = st.slider("Max depth", 5, 50, 20, 5)
            max_features = st.selectbox("Max features", ['sqrt', 'log2', None])
            
            if st.form_submit_button("Train Random Forest"):
                with st.spinner("Training Black Box Model..."):
                    try:
                        X_train, X_test, y_train, y_test = get_test_data()
                        
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            max_features=max_features,
                            random_state=42,
                            n_jobs=-1,
                            verbose=1
                        )
                        model.fit(X_train, y_train)
                        
                        with open('blackBox_file.pkl', 'wb') as f:
                            pickle.dump(model, f)
                        
                        # Evaluation
                        y_pred = model.predict(X_test)
                        bal_acc = balanced_accuracy_score(y_test, y_pred)
                        
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>✅ Training Complete</h4>
                        <p>Black Box model trained successfully!</p>
                        <div class="metric-box">
                            <b>Test Balanced Accuracy:</b> {bal_acc:.2%}<br>
                            <b>Regular Accuracy:</b> {model.score(X_test, y_test):.2%}
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Reload models
                        models = load_models()
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
    
    with col2:
        st.subheader("GAN Components")
        st.markdown("""
        <p>Generator and Substitute Detector forming the adversarial network.</p>
        """, unsafe_allow_html=True)
        
        with st.form("gan_config"):
            gen_layers = st.text_input("Generator layers", "128, 128")
            det_layers = st.text_input("Detector layers", "128, 64")
            learning_rate = st.selectbox("Learning rate", [0.001, 0.01, 0.1])
            activation = st.selectbox("Activation", ['relu', 'tanh', 'logistic'])
            
            if st.form_submit_button("Train GAN Components"):
                with st.spinner("Training GAN Components..."):
                    try:
                        X_train, X_test, y_train, y_test = get_test_data()
                        
                        # Scale data
                        scaler = MinMaxScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Save scaler
                        with open('scaler.pkl', 'wb') as f:
                            pickle.dump(scaler, f)
                        
                        # Parse layer sizes
                        gen_layer_sizes = tuple(map(int, gen_layers.split(',')))
                        det_layer_sizes = tuple(map(int, det_layers.split(',')))
                        
                        # Generator
                        generator = MLPRegressor(
                            hidden_layer_sizes=gen_layer_sizes,
                            activation=activation,
                            random_state=42,
                            learning_rate_init=learning_rate,
                            max_iter=500,
                            early_stopping=True,
                            verbose=True
                        )
                        
                        # Train generator (noise -> real samples)
                        noise = np.random.normal(0, 1, X_train_scaled.shape)
                        generator.fit(noise, X_train_scaled)
                        
                        # Substitute Detector
                        substituteDetector = MLPClassifier(
                            hidden_layer_sizes=det_layer_sizes,
                            activation=activation,
                            random_state=42,
                            learning_rate_init=learning_rate,
                            max_iter=500,
                            early_stopping=True,
                            verbose=True
                        )
                        substituteDetector.fit(X_train_scaled, y_train)
                        
                        # Save models
                        with open('generator.pkl', 'wb') as f:
                            pickle.dump(generator, f)
                        with open('substituteDetector.pkl', 'wb') as f:
                            pickle.dump(substituteDetector, f)
                        
                        # Evaluation
                        y_pred = substituteDetector.predict(X_test_scaled)
                        bal_acc = balanced_accuracy_score(y_test, y_pred)
                        
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>✅ Training Complete</h4>
                        <p>GAN components trained successfully!</p>
                        <div class="metric-box">
                            <b>Generator Architecture:</b> {gen_layer_sizes}<br>
                            <b>Detector Architecture:</b> {det_layer_sizes}<br>
                            <b>Test Balanced Accuracy:</b> {bal_acc:.2%}<br>
                            <b>Regular Accuracy:</b> {substituteDetector.score(X_test_scaled, y_test):.2%}
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Reload models
                        models = load_models()
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

# Page 4: Black Box Model
elif page == "Black Box Model":
    st.title("Black Box Detector Analysis")
    
    if not validate_model(models['blackBox'], "Black Box"):
        st.error("Black Box model not found or invalid. Please train the model first.")
        st.stop()
    
    st.subheader("Model Information")
    st.write(f"Model type: {type(models['blackBox']).__name__}")
    st.write(f"Number of trees: {models['blackBox'].n_estimators}")
    
    # Prepare test data
    X_train, X_test, y_train, y_test = get_test_data()
    
    st.subheader("Feature Importance")
    importances = models['blackBox'].feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_df.head(20), x='Importance', y='Feature', 
                title="Top 20 Important Features",
                color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Performance Metrics")
    y_pred = models['blackBox'].predict(X_test)
    y_proba = models['blackBox'].predict_proba(X_test)[:,1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Classification Report**")
        st.code(classification_report(y_test, y_pred, target_names=['Benign', 'Malware']))
    
    with col2:
        st.markdown("**Key Metrics**")
        st.metric("Balanced Accuracy", f"{balanced_accuracy_score(y_test, y_pred):.2%}")
        st.metric("Regular Accuracy", f"{models['blackBox'].score(X_test, y_test):.2%}")
    
    st.subheader("Confusion Matrix")
    fig = px.imshow(confusion_matrix(y_test, y_pred),
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Benign', 'Malware'],
                   y=['Benign', 'Malware'],
                   text_auto=True,
                   aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

# Page 5: GAN Components
elif page == "GAN Components":
    st.title("GAN Components Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generator Network")
        if not validate_model(models['generator'], "Generator"):
            st.error("Generator not trained or invalid. Please train first.")
        else:
            st.write(f"**Type:** {type(models['generator']).__name__}")
            st.write(f"**Architecture:** {models['generator'].hidden_layer_sizes}")
            st.write(f"**Activation:** {models['generator'].activation}")
            
            if st.button("Generate Sample"):
                try:
                    noise = np.random.normal(0, 1, (1, dataframe.shape[1]-1))
                    sample = models['generator'].predict(noise)
                    
                    st.write("**Generated sample features:**")
                    sample_df = pd.DataFrame(sample, columns=dataframe.columns[:-1])
                    st.dataframe(sample_df.style.format("{:.4f}"))
                    
                    # Visualize feature values
                    fig = px.bar(sample_df.T, title="Generated Sample Feature Values")
                    fig.update_layout(showlegend=False, xaxis_title="Feature", yaxis_title="Value")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
    
    with col2:
        st.subheader("Substitute Detector")
        if not validate_model(models['substituteDetector'], "Substitute Detector"):
            st.error("Substitute Detector not trained or invalid. Please train first.")
        else:
            st.write(f"**Type:** {type(models['substituteDetector']).__name__}")
            st.write(f"**Architecture:** {models['substituteDetector'].hidden_layer_sizes}")
            st.write(f"**Activation:** {models['substituteDetector'].activation}")
            
            # Prepare test data
            X_train, X_test, y_train, y_test = get_test_data()
            
            # Scale data
            if not hasattr(models['scaler'], 'scale_'):
                models['scaler'].fit(X_train)
            X_test_scaled = models['scaler'].transform(X_test)
            
            sample_idx = st.slider("Select test sample", 0, len(X_test)-1, 0)
            sample = X_test_scaled[sample_idx:sample_idx+1]
            sample_pred = models['substituteDetector'].predict(sample)[0]
            sample_proba = models['substituteDetector'].predict_proba(sample)[0]
            actual_label = y_test.iloc[sample_idx]
            
            st.markdown(f"""
            **Sample Prediction:**  
            {"⚠️ Malware" if sample_pred else "✅ Benign"}  
            (Actual: {"⚠️ Malware" if actual_label else "✅ Benign"})
            """)
            
            # Probability visualization
            proba_df = pd.DataFrame({
                'Class': ['Benign', 'Malware'],
                'Probability': sample_proba
            })
            fig = px.bar(proba_df, x='Class', y='Probability', 
                        color='Class', color_discrete_map={'Benign':'green', 'Malware':'red'},
                        text='Probability', range_y=[0,1])
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# Page 6: Evaluation
elif page == "Evaluation":
    st.title("System Evaluation")
    
    if not all(validate_model(m) for m in [models['blackBox'], models['substituteDetector']]):
        st.error("Required models not found or invalid. Please train all models first.")
        st.stop()
    
    st.subheader("Model Comparison")
    
    # Prepare test data
    X_train, X_test, y_train, y_test = get_test_data()
    
    # Scale data for substitute detector
    if not hasattr(models['scaler'], 'scale_'):
        models['scaler'].fit(X_train)
    X_test_scaled = models['scaler'].transform(X_test)
    
    # Get predictions
    bb_pred = models['blackBox'].predict(X_test)
    bb_proba = models['blackBox'].predict_proba(X_test)[:,1]
    sd_pred = models['substituteDetector'].predict(X_test_scaled)
    sd_proba = models['substituteDetector'].predict_proba(X_test_scaled)[:,1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Black Box Detector")
        st.code(classification_report(y_test, bb_pred, target_names=['Benign', 'Malware']))
        
        fig = px.imshow(confusion_matrix(y_test, bb_pred),
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Benign', 'Malware'],
                       y=['Benign', 'Malware'],
                       text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Substitute Detector")
        st.code(classification_report(y_test, sd_pred, target_names=['Benign', 'Malware']))
        
        fig = px.imshow(confusion_matrix(y_test, sd_pred),
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Benign', 'Malware'],
                       y=['Benign', 'Malware'],
                       text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Performance Comparison")
    metrics = {
        'Model': ['Black Box', 'Substitute'],
        'Balanced Accuracy': [
            balanced_accuracy_score(y_test, bb_pred),
            balanced_accuracy_score(y_test, sd_pred)
        ],
        'Precision (Malware)': [
            precision_score(y_test, bb_pred),
            precision_score(y_test, sd_pred)
        ],
        'Recall (Malware)': [
            recall_score(y_test, bb_pred),
            recall_score(y_test, sd_pred)
        ]
    }

    # Convert to DataFrame and format
    metrics_df = pd.DataFrame(metrics)
    styled_df = metrics_df.style.format({
        'Balanced Accuracy': '{:.2%}',
        'Precision (Malware)': '{:.2%}',
        'Recall (Malware)': '{:.2%}'
    }).highlight_max(axis=0)

    st.dataframe(styled_df)

# Page 7: Adversarial Attack
elif page == "Adversarial Attack":
    st.title("Adversarial Example Generation")
    
    # Debug: Check which models are missing
    missing_models = []
    if not validate_model(models['blackBox']):
        missing_models.append("Black Box Model")
    if not validate_model(models['generator']):
        missing_models.append("Generator")
    if not validate_model(models['substituteDetector']):
        missing_models.append("Substitute Detector")
    if not hasattr(models['scaler'], 'scale_'):
        missing_models.append("Scaler (not fitted)")
    
    if missing_models:
        st.error(f"❌ Missing/Invalid Models: {', '.join(missing_models)}")
        st.error("Please train all models first in the 'Model Training' section.")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <h4>Adversarial Attack Simulation</h4>
    <p>Generate samples that bypass the black box detector while maintaining malicious functionality.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data
    X_train, X_test, y_train, y_test = get_test_data()
    
    # Select malware sample to attack
    malware_samples = X_test[y_test == 1]
    if len(malware_samples) == 0:
        st.error("No malware samples available for testing")
        st.stop()
    
    sample_idx = st.selectbox("Select malware sample", 
                            options=range(len(malware_samples)),
                            format_func=lambda x: f"Sample {x}")
    original_sample = malware_samples.iloc[sample_idx:sample_idx+1]
    
    st.subheader("Original Sample")
    # Safe formatting for mixed data types
    numeric_cols = original_sample.select_dtypes(include=[np.number]).columns
    styled_original = original_sample.style.format("{:.4f}", subset=numeric_cols)
    st.dataframe(styled_original)
    
    # Original prediction
    original_pred = models['blackBox'].predict(original_sample)[0]
    original_proba = models['blackBox'].predict_proba(original_sample)[0][original_pred]
    st.markdown(f"""
    **Original Prediction:** {"⚠️ Malware" if original_pred else "✅ Benign"}  
    **Confidence:** {original_proba:.1%}
    """)
    
    st.subheader("Attack Parameters")
    attack_strength = st.slider("Perturbation strength", 0.1, 2.0, 0.5, 0.1)
    noise_scale = st.slider("Noise scale", 0.1, 2.0, 1.0, 0.1)
    
    if st.button("Generate Adversarial Example"):
        with st.spinner("Crafting adversarial sample..."):
            try:
                # Generate perturbation
                noise = np.random.normal(0, noise_scale, original_sample.shape)
                perturbation = models['generator'].predict(noise) * attack_strength
                
                # Create adversarial example (convert to numpy array for operations)
                adversarial = original_sample.values + perturbation
                adversarial = np.clip(adversarial, 0, 1)  # Keep features valid
                
                # Get predictions (convert back to DataFrame for prediction)
                adversarial_df = pd.DataFrame(adversarial, columns=original_sample.columns)
                adversarial_pred = models['blackBox'].predict(adversarial_df)[0]
                adversarial_proba = models['blackBox'].predict_proba(adversarial_df)[0][adversarial_pred]
                
                st.subheader("Adversarial Sample")
                # Safe formatting for adversarial sample
                styled_adversarial = pd.DataFrame(adversarial, columns=original_sample.columns).style.format("{:.4f}", subset=numeric_cols)
                st.dataframe(styled_adversarial)
                
                st.markdown(f"""
                **Adversarial Prediction:** {"✅ Benign" if not adversarial_pred else "⚠️ Malware"}  
                **Confidence:** {adversarial_proba:.1%}
                """)
                
                # Show feature changes
                changes = (adversarial - original_sample.values).flatten()
                change_df = pd.DataFrame({
                    'Feature': original_sample.columns,
                    'Change': changes,
                    'Absolute Change': np.abs(changes)
                }).sort_values('Absolute Change', ascending=False)
                
                st.write("Top Feature Changes:")
                # Format only numeric columns in change_df
                numeric_change_cols = change_df.select_dtypes(include=[np.number]).columns
                styled_changes = change_df.style.format("{:.4f}", subset=numeric_change_cols)\
                    .bar(subset=['Change'], align='mid', color=['#d65f5f', '#5fba7d'])
                st.dataframe(styled_changes)
                
                # Visualize the changes
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Original", "Adversarial"))
                
                fig.add_trace(
                    go.Bar(x=original_sample.columns, y=original_sample.values.flatten(), name="Original"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=original_sample.columns, y=adversarial.flatten(), name="Adversarial"),
                    row=1, col=2
                )
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                if not adversarial_pred:
                    st.success("✅ Attack Successful - Sample classified as benign!")
                else:
                    st.warning("⚠️ Attack Unsuccessful - Try increasing perturbation strength")
                
            except Exception as e:
                st.error(f"Attack generation failed: {str(e)}")            
# Footer
st.markdown("""
---
**MalGAN Dashboard** | *For research purposes only* | Made by ghostvirus
""")