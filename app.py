from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import re # Tambahkan modul regex

app = Flask(__name__)

# --- KONFIGURASI ---
MODEL_RMSE = 22000 # RMSE dari Ridge Regression

print("Loading models...")
try:
    # Load semua aset model termasuk Scaler
    model = joblib.load('models/model_linreg.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf.pkl')
    model_columns = joblib.load('models/feature_columns.pkl')
    scaler = joblib.load('models/scaler.pkl') 
    
    available_cities = sorted([col.replace('city_', '') for col in model_columns if col.startswith('city_')])
    print("Models & Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    scaler = None
    available_cities = []
    model_columns = [] # Inisialisasi list kosong jika gagal load

# --- FUNGSI CHART ---
def generate_shap_plot(input_df, model, columns):
    # input_df di sini sudah dalam bentuk scaled
    coefficients = model.coef_
    input_values = input_df.values[0]
    raw_impacts = coefficients * input_values
    
    # Inisialisasi Kategori
    grouped_impacts = {'Nama Produk': 0, 'Lokasi Toko': 0, 'Harga': 0, 'Diskon': 0, 'Rating': 0, 'Jml Ulasan': 0}
    
    for feature, impact in zip(columns, raw_impacts):
        # 1. Nama Produk
        if 'tfidf_' in feature or 'name_len' in feature:
            grouped_impacts['Nama Produk'] += impact
        
        # 2. Lokasi
        elif 'city_' in feature:
            grouped_impacts['Lokasi Toko'] += impact
        
        # 3. Harga
        elif 'original_price' in feature or 'final_price' in feature or 'price_clean' in feature: 
            grouped_impacts['Harga'] += impact
        
        # 4. Diskon
        elif 'discount_clean' in feature:
            grouped_impacts['Diskon'] += impact
        
        # 5. Rating
        elif 'rating_clean' in feature:
            grouped_impacts['Rating'] += impact
        
        # 6. Ulasan
        elif 'review_count_clean' in feature:
            grouped_impacts['Jml Ulasan'] += impact
            
    df_plot = pd.DataFrame(list(grouped_impacts.items()), columns=['Fitur', 'Impact']).sort_values(by='Impact', ascending=True)

    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid') 
    colors = ['#ef4444' if x < 0 else '#10b981' for x in df_plot['Impact']]
    
    bars = plt.barh(df_plot['Fitur'], df_plot['Impact'], color=colors, alpha=0.9)
    plt.xlabel('Kontribusi (Scaled)', fontsize=10, fontweight='bold', color='#374151')
    plt.title('Analisis Faktor Penentu', fontsize=12, fontweight='bold', color='#111827', loc='left')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Label pada bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else width
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f' {int(width):,}', va='center', fontsize=9, color='#374151')

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\b\d+\w*\b', '', text) # Hapus angka/satuan seperti 100gr
    return text

# --- ROUTE UTAMA ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    prediction_range = None
    plot_url = None
    warnings = []
    confidence_score = 0
    
    # Default Values Form HTML
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
            
            # --- 1. AMBIL INPUT USER ---
            name_input = request.form.get('name', '')
            original_price_input = int(request.form.get('price', 0))
            discount_input = float(request.form.get('discount', 0))
            rating_input = float(request.form.get('rating', 0))
            review_count_input = int(request.form.get('reviews', 0))
            city_input = request.form.get('city', '')

            # --- 2. VALIDASI INPUT ---
            if len(name_input) < 10: warnings.append("Nama produk terlalu pendek.")
            if original_price_input < 1000: warnings.append("Harga di bawah Rp1.000 tidak wajar.")
            if rating_input == 5.0 and review_count_input > 50: warnings.append("Rating 5.0 sempurna dengan banyak ulasan mencurigakan.")

            # --- 3. PREPROCESSING (LOGIC UTAMA) ---
            
            # A. Hitung Final Price
            final_price_calculated = original_price_input * (1 - (discount_input / 100))

            # B. Siapkan DataFrame Input (Sesuai Kolom Model)
            input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
            
            # C. Isi Data Mentah ke DataFrame
            if 'original_price' in input_data.columns: input_data['original_price'] = original_price_input
            if 'final_price' in input_data.columns: input_data['final_price'] = final_price_calculated
            if 'discount_clean' in input_data.columns: input_data['discount_clean'] = discount_input
            if 'rating_clean' in input_data.columns: input_data['rating_clean'] = rating_input
            if 'review_count_clean' in input_data.columns: input_data['review_count_clean'] = review_count_input
            if 'name_len' in input_data.columns: input_data['name_len'] = len(name_input)

            # D. PENERAPAN STANDARD SCALER
            numeric_cols = ['final_price', 'original_price', 'discount_clean', 'rating_clean', 'review_count_clean', 'name_len']
            
            if scaler:
                input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
            
            # E. TF-IDF Transform (Teks)
            if tfidf_vectorizer:
                # Bersihkan teks terlebih dahulu
                name_clean = clean_text(name_input)
                tfidf_matrix = tfidf_vectorizer.transform([name_clean]).toarray()
                
                # Gunakan get_feature_names_out() untuk mapping yang benar
                feature_names = tfidf_vectorizer.get_feature_names_out()
                
                for i, word in enumerate(feature_names):
                    col_name = f'tfidf_{word}' # Sesuaikan format nama kolom
                    if col_name in input_data.columns: 
                        input_data[col_name] = tfidf_matrix[0, i]

            # F. One-Hot Encoding Lokasi
            city_col = f'city_{city_input}'
            if city_col in input_data.columns: 
                input_data[city_col] = 1
            else:
                 # Jika kota tidak ada di list model, masukkan ke 'Other' jika ada
                if 'city_Other' in input_data.columns:
                    input_data['city_Other'] = 1

            # --- 4. PREDIKSI ---
            if model:
                raw_prediction = model.predict(input_data)[0]
                
                lower_bound = max(0, int(raw_prediction - MODEL_RMSE))
                upper_bound = int(raw_prediction + MODEL_RMSE)
                predicted_sales = max(0, int(raw_prediction))

                prediction_text = f"{predicted_sales:,}"
                prediction_range = f"{lower_bound:,} - {upper_bound:,}"
                
                plot_url = generate_shap_plot(input_data, model, model_columns)

                # Score Keyakinan
                base_score = 95
                if warnings: base_score -= (len(warnings) * 15)
                if city_input not in available_cities: base_score -= 10
                confidence_score = max(10, min(99, base_score))

        except Exception as e:
            warnings.append(f"Terjadi kesalahan sistem: {str(e)}")
            print(f"DEBUG ERROR: {e}") 

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