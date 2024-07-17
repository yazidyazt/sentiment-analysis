from flask import Flask, render_template, request
from predict import predict_comment, tokenizer
import logging

app = Flask(__name__)

# Konfigurasikan logging untuk debugging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def prediction():
    comment = request.form['comment']
    categories = ['Netral', 'Positif', 'Negatif']  # Sesuaikan dengan kategori Anda
    pred_prob, predicted_category, pred_labels = predict_comment(tokenizer, comment, categories)
    
    # Log hasil prediksi
    logging.info(f"Comment: {comment}")
    logging.info(f"Predicted Category: {predicted_category}")
    logging.info(f"Prediction Probabilities: {pred_prob}")
    logging.info(f"Prediction Labels: {pred_labels}")
    
    # Setiap kategori diberi nilai 0 atau 1
    netral = 'True' if pred_labels == 0 else ''
    positif = 'True' if pred_labels == 1 else ''
    negatif = 'True' if pred_labels == 2 else ''
    
    # Tampilkan hasil prediksi
    return render_template('main.html', comment=comment, predicted_category=predicted_category, netral=netral, positif=positif, negatif=negatif)

if __name__ == '__main__':
    app.run(debug=True)

