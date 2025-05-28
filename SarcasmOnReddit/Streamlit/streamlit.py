import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer(model_path, vectorizer_path):
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure the trained model exists.")
        return None, None
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer file '{vectorizer_path}' not found. Please ensure the vectorizer file exists.")
        return None, None
    try:
        nb_clf = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        return nb_clf, tfidf_vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {str(e)}")
        return None, None

# Define label mapping
label_map = {
    0: "Sarcasm",
    1: "Not Sarcasm",
    2: "Unclear",
    3: "Others"
}

def display_prediction_details(model, vectorizer, input_text, label_map):
    st.subheader("🔍 Prediction Results")
    try:
        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)[0]
        probs = model.predict_proba(input_vector)[0]
        pred_label = label_map.get(prediction, str(prediction))
        confidence = probs[prediction]

        st.markdown(f"### 🧠 Predicted Class: *{pred_label}*")
        st.markdown(f"*Confidence:* {confidence:.2%}")

        # Display table and bar chart
        st.markdown("### 📊 Class Probabilities")

        prob_df = pd.DataFrame({
            'Class': [label_map[i] for i in range(len(probs))],
            'Probability': probs
        }).sort_values(by='Probability', ascending=False)

        # Show probabilities in text
        for _, row in prob_df.iterrows():
            st.write(f"**{row['Class']}**: {row['Probability']:.2%}")

        # Show chart
        st.bar_chart(prob_df.set_index('Class'))

        with st.expander("🔍 See Detailed Explanation (LIME)"):
            explain_prediction_with_lime(model, vectorizer, input_text, label_map)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

def explain_prediction_with_lime(model, vectorizer, input_text, label_map):
    class_names = [label_map[i] for i in sorted(label_map)]
    pipeline = make_pipeline(vectorizer, model)
    explainer = LimeTextExplainer(class_names=class_names)

    try:
        lime_exp = explainer.explain_instance(
            input_text,
            pipeline.predict_proba,
            num_features=8,
            labels=[0, 1, 2, 3]
        )

        label_to_plot = pipeline.predict([input_text])[0]
        word_contributions = dict(lime_exp.as_list(label=label_to_plot))

        st.subheader("🧠 Word Contributions (LIME Explanation)")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['green' if v > 0 else 'red' for v in word_contributions.values()]
            ax.barh(list(word_contributions.keys()), word_contributions.values(), color=colors)
            ax.set_xlabel("Contribution", fontsize=9)
            ax.set_title(f"Class: {label_map[label_to_plot]}", fontsize=11)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")

def show_model_info():
    st.subheader("📌 Model Info")
    st.markdown("""
    *Model:* Political Sarcasm Detector v1.0  
    *Description:* This model is designed to classify text related to political discourse into four categories, focusing on detecting sarcasm in political contexts using a Naive Bayes classifier with Bag-of-Words features.  
    *Classes:*  
    - 🟢 *Sarcasm*: Text with sarcastic intent, often mocking political figures, policies, or events  
    - ✅ *Not Sarcasm*: Sincere political statements or opinions without sarcastic intent  
    - ❓ *Unclear*: Political text where sarcasm is ambiguous or context-dependent  
    - 🔄 *Others*: Text unrelated to sarcasm or politics, or neutral political statements  
    """)
    st.markdown("*Note:* Performance metrics are not available without a dataset.")

def show_example_predictions():
    st.subheader("💡 Example Predictions")
    examples = [
        ("Wow, another brilliant policy from our flawless leaders 🙄", "Sarcasm"),
        ("I support the new healthcare reform; it’s a step forward!", "Not Sarcasm"),
        ("Is this politician serious about that promise?", "Unclear"),
        ("The meeting is scheduled for 3 PM tomorrow.", "Others")
    ]
    for text, label in examples:
        st.markdown(f"- *Input:* {text}")
        st.markdown(f"  *Predicted Class:* {label}")

def show_limitations():
    st.subheader("⚠ Caveats & Limitations")
    st.markdown("""
    - The model may misinterpret sarcasm in political texts due to complex rhetoric or lack of context.
    - Political jargon, regional references, or culture-specific humor may lead to misclassification.
    - Trained primarily on English political texts, so sarcasm in multilingual or non-political contexts may not be detected accurately.
    - Subtle irony or satire in political discourse may be classified as 'Unclear' or 'Others' without clear sarcastic markers.
    """)

def main():
    st.set_page_config(page_title="Political Sarcasm Detector", layout="wide")
    st.title("🧪 Political Sarcasm Detector Dashboard")

    model_path = "NB_BoW.pkl"
    vectorizer_path = "bow_vectorizer.pkl"

    nb_clf, tfidf_vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    
    if nb_clf is None or tfidf_vectorizer is None:
        st.error("Cannot proceed without a valid model and vectorizer.")
        return

    tab1, tab2, tab3 = st.tabs(["🔍 Detect Sarcasm", "📊 Model Info", "📚 Examples & Limitations"])

    with tab1:
        st.markdown("### 🎯 Enter text to detect sarcasm in political context:")
        user_input = st.text_area("Input Text", placeholder="Type or paste your political text here...", height=180)
        if st.button("Analyze Text"):
            if user_input.strip():
                display_prediction_details(nb_clf, tfidf_vectorizer, user_input, label_map)
            else:
                st.warning("Please enter some text before analyzing.")

    with tab2:
        show_model_info()

    with tab3:
        show_example_predictions()
        show_limitations()

if __name__ == "__main__":
    main()