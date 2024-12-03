from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Khởi tạo Flask app
app = Flask(__name__)

# Tải các mô hình và dữ liệu đã lưu
loaded_model = load_model('D:/Document/doan2/model.h5')

with open('D:/Document/doan2/user_id_mapping.pkl', 'rb') as f_user:
    user_id_mapping_load = pickle.load(f_user)

with open('D:/Document/doan2/product_id_mapping.pkl', 'rb') as f_product:
    product_id_mapping_load = pickle.load(f_product)

features_scaled = joblib.load('D:/Document/doan2/features_scaled.pkl', mmap_mode='r')
df_product = pd.read_pickle('D:/Document/doan2/df_product.pkl')

product_info = pd.read_csv(r'D:\Document\doan2\data-doan2-clean\product_clean.csv')

# Hàm tính độ tương tự sản phẩm
def get_single_product_similarity(product_id):
    if product_id not in df_product['product_id'].values:
        return "Product ID not found."
    product_index = df_product.index[df_product['product_id'] == product_id].tolist()[0]
    similarity_scores = cosine_similarity(features_scaled[product_index].reshape(1, -1), features_scaled)
    return pd.Series(similarity_scores.flatten(), index=df_product['product_id'])

# Hàm gợi ý sản phẩm
def get_product_recommendations(product_id, num_recommendations=5):
    similarity_scores = get_single_product_similarity(product_id)
    recommended_products = similarity_scores.sort_values(ascending=False).index[1:num_recommendations + 1]
    return df_product[df_product['product_id'].isin(recommended_products)]

# API endpoint để nhận thông tin sản phẩm gợi ý dưới dạng JSON
@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    try:
        # Nhận dữ liệu từ query parameters
        product_id = request.args.get('product_id', type=int) 
        user_id = request.args.get('user_id', type=int)
        num_recommendations = request.args.get('num_recommendations', default=50, type=int) 

        # Kiểm tra nếu không có tham số product_id hoặc user_id
        if not product_id or not user_id:
            return jsonify({"error": "product_id and user_id are required."}), 400

        # Gợi ý sản phẩm
        recommended_products = get_product_recommendations(product_id=product_id, num_recommendations=num_recommendations)
        
        # Thực hiện ánh xạ và dự đoán ratings
        product_ids = recommended_products['product_id'].values
        user_idx = user_id_mapping_load.get(user_id)

        if user_idx is None:
            return jsonify({"error": "User ID không tồn tại trong ánh xạ."}), 400
        
        valid_product_indices = []
        valid_product_ids = []

        for pid in product_ids:
            product_idx = product_id_mapping_load.get(pid)
            if product_idx is not None:
                valid_product_indices.append(product_idx)
                valid_product_ids.append(pid)

        if not valid_product_indices:
            return jsonify({"error": "Không có Product ID nào hợp lệ trong ánh xạ."}), 400

        user_input = np.array([[user_idx]] * len(valid_product_indices))
        product_input = np.array(valid_product_indices).reshape(-1, 1) 
        predicted_ratings = loaded_model.predict([user_input, product_input])
        
        product_rating_pairs = list(zip(valid_product_ids, predicted_ratings.flatten()))
        
        # Lọc các sản phẩm theo rating >= 3 và không có review
        filtered_product_ids_by_rating = [pid for pid, rating in product_rating_pairs if rating >= 3]
        filtered_product_ids_by_review_count = recommended_products[recommended_products['review_count'] == 0]['product_id'].values.tolist()
        filtered_product_ids = filtered_product_ids_by_rating + filtered_product_ids_by_review_count
        filtered_product_ids = list(set(filtered_product_ids))

        filtered_product_info = product_info[product_info['id'].isin(filtered_product_ids)].copy()
        product_rating_dict = dict(product_rating_pairs)
        filtered_product_info['predicted_rating'] = filtered_product_info['id'].map(product_rating_dict)
        filtered_product_info = filtered_product_info.sort_values(by='predicted_rating', na_position='last')

        # Chuyển DataFrame thành dictionary và trả về dưới dạng JSON
        return jsonify(filtered_product_info.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
