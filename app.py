from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# --- KONFIGURASI ---
MODEL_RMSE = 22000 # Menggunakan RMSE dari Ridge Regression

print("Loading models...")
try:
    model = joblib.load('models/model_linreg.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf.pkl')
    model_columns = joblib.load('models/feature_columns.pkl')
    available_cities = sorted([col.replace('city_', '') for col in model_columns if col.startswith('city_')])
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    available_cities = []

# --- FUNGSI CHART (UPDATED: Hapus grup Deskripsi) ---
def generate_shap_plot(input_df, model, columns):
    coefficients = model.coef_
    input_values = input_df.values[0]
    raw_impacts = coefficients * input_values
    
    # Hapus kategori 'Deskripsi'
    grouped_impacts = {'Nama Produk': 0, 'Lokasi Toko': 0, 'Harga': 0, 'Diskon': 0, 'Rating': 0, 'Jml Ulasan': 0}
    
    for feature, impact in zip(columns, raw_impacts):
        if 'tfidf_' in feature or 'name_len' in feature: # desc_len dihapus
            grouped_impacts['Nama Produk'] += impact
        elif 'city_' in feature:
            grouped_impacts['Lokasi Toko'] += impact
        elif 'price_clean' in feature or 'final_price' in feature:
            grouped_impacts['Harga'] += impact
        elif 'discount_clean' in feature:
            grouped_impacts['Diskon'] += impact
        elif 'rating_clean' in feature:
            grouped_impacts['Rating'] += impact
        elif 'review_count_clean' in feature:
            grouped_impacts['Jml Ulasan'] += impact
            
    df_plot = pd.DataFrame(list(grouped_impacts.items()), columns=['Fitur', 'Impact']).sort_values(by='Impact', ascending=True)

    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid') 
    colors = ['#ef4444' if x < 0 else '#10b981' for x in df_plot['Impact']]
    
    bars = plt.barh(df_plot['Fitur'], df_plot['Impact'], color=colors, alpha=0.9)
    plt.xlabel('Kontribusi Unit', fontsize=10, fontweight='bold', color='#374151')
    plt.title('Analisis Faktor Penentu', fontsize=12, fontweight='bold', color='#111827', loc='left')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else width
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f' {int(width):,}', va='center', fontsize=9, color='#374151')

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# --- ROUTE UTAMA ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    prediction_range = None
    plot_url = None
    warnings = []
    confidence_score = 0
    
    # 1. Default Values (Deskripsi DIHAPUS)
    form_data = {
        'name': '', 
        'price': '', 
        'discount': 0, 
        'rating': 4.5, 
        'reviews': 50, 
        'city': ''
    }

    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            
            # Ambil Input (Deskripsi DIHAPUS)
            name_input = request.form.get('name', '')
            price_input = int(request.form.get('price', 0))
            discount_input = float(request.form.get('discount', 0))
            rating_input = float(request.form.get('rating', 0))
            review_count_input = int(request.form.get('reviews', 0))
            city_input = request.form.get('city', '')

            # --- Validasi ---
            if len(name_input) < 10: warnings.append("Nama produk terlalu pendek.")
            if price_input < 1000: warnings.append("Harga di bawah Rp1.000 tidak wajar.")
            if rating_input == 5.0 and review_count_input > 50: warnings.append("Rating 5.0 sempurna dengan banyak ulasan mencurigakan.")

            # --- Preprocessing ---
            input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
            input_data['price_clean'] = price_input
            input_data['discount_clean'] = discount_input
            input_data['rating_clean'] = rating_input
            input_data['review_count_clean'] = review_count_input
            input_data['final_price'] = price_input * (1 - (discount_input / 100))
            input_data['name_len'] = len(name_input)
            
            # [HAPUS] desc_len tidak lagi dihitung
            # input_data['desc_len'] = len(desc_input) 

            # TF-IDF Transform
            if tfidf_vectorizer:
                tfidf_matrix = tfidf_vectorizer.transform([name_input]).toarray()
                for i in range(tfidf_matrix.shape[1]):
                    col_name = f'tfidf_{i}'
                    if col_name in input_data.columns: input_data[col_name] = tfidf_matrix[0, i]

            # One-Hot Encoding Lokasi
            city_col = f'city_{city_input}'
            if city_col in input_data.columns: input_data[city_col] = 1

            # --- Prediksi ---
            if model:
                raw_prediction = model.predict(input_data)[0]
                
                lower_bound = max(0, int(raw_prediction - MODEL_RMSE))
                upper_bound = int(raw_prediction + MODEL_RMSE)
                predicted_sales = max(0, int(raw_prediction))

                prediction_text = f"{predicted_sales:,}"
                prediction_range = f"{lower_bound:,} - {upper_bound:,}"
                
                plot_url = generate_shap_plot(input_data, model, model_columns)

                # --- CONFIDENCE SCORE (LOGIKA BARU - Tanpa Deskripsi) ---
                base_score = 95
                
                if warnings:
                    base_score -= (len(warnings) * 15)
                
                if city_input not in available_cities:
                    base_score -= 10
                
                # [HAPUS] Penalti deskripsi pendek dihapus karena inputnya tidak ada
                
                confidence_score = max(10, min(99, base_score))

        except Exception as e:
            warnings.append(f"Terjadi kesalahan sistem: {str(e)}")

    return render_template('index.html', 
                           prediction=prediction_text, 
                           range=prediction_range,
                           confidence=confidence_score,
                           plot_url=plot_url,
                           cities=available_cities,
                           form=form_data, 
                           warnings=warnings)

if __name__ == '__main__':
    app.run(debug=True)