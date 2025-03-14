import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load trained model
pipe_lr = joblib.load(open("C:/Users/BHUMI/Downloads/classifier_emotions_model (1)", "rb"))

# Updated mapping of numerical labels to emotion names
emotion_labels = {
    0: "joy", 1: "sadness", 2: "anger", 3: "fear", 4: "love", 5: "surprise"
}

# Emoji dictionary for display
emotions_emoji_dict = {
    "joy": "😂", "sadness": "😔", "anger": "😠", "fear": "😨", 
    "love": "❤️", "surprise": "😮"
}

# Function to predict emotion
def predict_emotions(docx):
    predicted_label = pipe_lr.predict([docx])[0]  # Get numerical prediction
    return emotion_labels.get(predicted_label, "Unknown")  # Convert to emotion name

# Function to get probability distribution
def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# Streamlit App
def main():
    st.title("Text Emotion Detection 🎭")
    st.subheader("Analyze the emotion behind your text")

    with st.form(key='emotion_form'):
        raw_text = st.text_area("Enter your text here:")
        submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        col1, col2 = st.columns(2)

        # Predictions
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.write(f"**{prediction}** {emoji_icon}")
            st.write(f"Confidence: **{np.max(probability):.4f}**")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=[emotion_labels[i] for i in range(len(emotion_labels))])
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotion", "Probability"]

            # Bar chart visualization
            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='Emotion',
                y='Probability',
                color='Emotion'
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
