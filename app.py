from flask import Flask, request, jsonify
from flask_cors import CORS

from inference import InferenceModel
import traceback


app = Flask(__name__)
CORS(app)

try:
    model = InferenceModel(path_to_weights="save_model/model.safetensors", huggingface_model=True)
except Exception as e:
    print("❌ Lỗi khi load mô hình:")
    traceback.print_exc()
    model = None

@app.route('/pred', methods=['POST'])
def prediction():
    payload = request.get_json()

    # Lấy dữ liệu từ request
    context = payload.get('context', '')
    question = payload.get('question', '')

    # # In ra terminal
    # print("\n===== Nhận yêu cầu mới =====")
    # print(f"Context: {context}")
    # print(f"Question: {question}")

    # Gọi mô hình
    prediction = model.inference_model(question, context)
    answer = prediction["answer"]

    return jsonify({"answer": answer}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)


# Chayj server
