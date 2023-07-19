from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

# Flask uygulamasını oluştur
app = Flask(__name__)

# Modelin yüklenmesi
model = tf.keras.models.load_model('model/MobileNetV2_model4.h5')
class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

calories_list = [237, 287, 316, 160, 200, 62, 343, 393, 285, 277, 137, 94, 340, 157, 421, 53, 321, 402, 158, 232, 203, 376, 246, 398, 148, 231, 135, 345, 373, 305, 158, 452, 289, 121, 129, 250, 333, 250, 274, 459, 312, 157, 188, 201, 111, 159, 296, 145, 118, 407, 206, 160, 271, 250, 73, 296, 116, 166, 207, 150, 163, 290, 310, 449, 36, 172, 268, 154, 407, 68, 205, 160, 227, 169, 339, 114, 266, 250, 401, 320, 252, 111, 150, 350, 120, 190, 63,111, 50, 105, 140, 235, 190, 250, 250, 130, 267, 122, 370, 200, 291]


# Anasayfa
@app.route('/')
def home():
    return render_template('indexx.html')

# Görüntüyü sınıflandırma
@app.route('/classify', methods=['POST'])
def classify():
    # Kullanıcıdan gönderilen görüntüyü al
    file = request.files['image']

    # Görüntüyü yükle ve yeniden boyutlandır
    img = Image.open(file)
    img = img.resize((224, 224))

    # Görüntüyü diziye dönüştür
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Sınıflandırma yap
    predictions = model.predict(img_array)
    predicted_class_indices = np.argmax(predictions, axis=1)
    predicted_classes = [class_names[idx] for idx in predicted_class_indices]
    predicted_calorie = [calories_list[idx] for idx in predicted_class_indices]

    # Sonucu döndür
    return render_template('result.html', predicted_classes=predicted_classes, predicted_calorie=predicted_calorie)


# Uygulamayı çalıştır
if __name__ == '__main__':
    app.run(debug=True)
