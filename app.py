import os
import uuid
from tensorflow import keras
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(os.path.join(BASE_DIR, 'model.hdf5'))
ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])
classes = ['Cyst', 'Normal', 'Stone', 'Tumor', 'NO Kidney']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def predict(filename, model):
    img = load_img(filename, target_size=(200, 200), color_mode='grayscale')
    img = img_to_array(img)
    img = img.reshape(1, 200, 200, 1)
    img = img.astype('float32')
    img = img / 255
    result = model.predict(img)

    dict_result = {}
    for i in range(min(len(classes), len(result[0]))):
        dict_result[result[0][i]] = classes[i]
    
    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:min(len(classes), len(res))]
    prob_result = []
    class_result = []
    for i in range(len(prob)):
        prob_result.append((prob[i] * 100).round(2))
        class_result.append(dict_result[prob[i]])

    # If the highest probability is less than 70%, classify as 'NO Kidney'
    if prob_result[0] < 70:
        class_result = ['NO Kidney'] + class_result[:4]
        prob_result = [100.0 - sum(prob_result)] + prob_result[:4]

    # Ensure class_result and prob_result have exactly 5 elements
    while len(class_result) < 5:
        class_result.append('NO Kidney')
    while len(prob_result) < 5:
        prob_result.append(0.0)

    return class_result, prob_result

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_image():
    target_img = os.path.join(BASE_DIR, 'static/images')
    if request.method == 'POST':
        if request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                file.save(img_path)
                img = filename
                class_result, prob_result = predict(img_path, model)
                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "class4": class_result[3],
                    "class5": class_result[4],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                    "prob4": prob_result[3],
                    "prob5": prob_result[4],
                }
                return render_template('success.html', img=img, predictions=predictions)
            else:
                error = "Please upload images of jpg, jpeg, and png extension only"
                return render_template('index.html', error=error)
            return render_template('index.html')
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)


