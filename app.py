import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Load the preprocessing steps and the model
encoder = joblib.load('model/En.pkl')
scaler = joblib.load('model/s.pkl')
model = joblib.load('model/rf_model.pkl')

def preprocess(data):
    df = pd.DataFrame([data])
    encoding_cols = ["Gender", "Parental Education Level", "Lunch Type", "Test Preparation Course"]
    scaling_cols = ["Study Time", "Absences"]
    
    # for col in encoding_cols:
    #     df[col] = df[col].str.strip()

    print(df.head())
    
    for col in encoding_cols:
        if col in encoder:
            le = encoder[col]
            df[col] = le.transform(df[col])

    df[scaling_cols] = scaler.transform(df[scaling_cols])
        
    return df

def predict(gender, PEL, lunch_type, TPC, study_time, absences):
    data = {
        "Gender": gender,
        "Parental Education Level": PEL,
        "Lunch Type": lunch_type,
        "Test Preparation Course": TPC,
        "Study Time": study_time,
        "Absences": absences
    }
    raw_data = preprocess(data)
    result = model.predict(raw_data)
    print(result)
    return result[0]  # Return the first (and only) prediction value

inputs = [

    gr.Radio(label="Gender", choices=["Female", "Male"]),
    gr.Dropdown(
            ["Associate", "Bachelor", "High School", "Master", "Some College"], label="Parental Education Level"),
    gr.Dropdown(
            ["Standard", "Free/Reduced"], label="Launch Type"),
    
    gr.Dropdown(
            ["Completed", "Not Completed"], label="Test Preparation Course"),
    gr.Number(label="Study Time"),
    gr.Number(label="Absences")
    
]

outputs = gr.Number(label="Prediction")

gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Student Exam Performance").launch()