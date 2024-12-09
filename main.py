import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

st.title("Класифікація зображення")
 
#Класи передбачень
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

#Вибір моделі
selected = option_menu(menu_title=None, options=["Simple RNN", "RNN with VGG16"])

#Функція для обробки зображення
def preprocess_image(image):
    image = image.resize((28, 28))  
    image = image.convert("L")  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=-1)  
    image = np.expand_dims(image, axis=0)  
    return image
def preprocess_image_vgg(image):
    image = image.resize((32, 32)) 
    image = image.convert("RGB") 
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)
    return image

#Функція для графіків
def graph(file):
    with open(file, 'r') as f:
            history = json.load(f)
    fig = plt.figure()
    epochs = range(1, len(history['accuracy']) + 1)
    plt.plot(epochs, history['accuracy'], "bo", label="Training acc")
    plt.plot(epochs, history['val_accuracy'], "b", label="Validation loss")
    plt.legend()
    st.pyplot(fig)
    fig2 = plt.figure()
    plt.plot(epochs, history['loss'], "bo", label="Training loss")
    plt.plot(epochs, history['val_loss'], "b", label="Validation loss")
    plt.legend()
    st.pyplot(fig2)

#Функція для передбачень
def pred(model, processed_image):
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]

    st.write(f"Передбачений клас: {predicted_label}")
    st.write(f"Передбачення для класів: {predictions}")


#Завантаження файлу     
uploaded_file = st.file_uploader("Виберіть зображення...", type=["jpg", "jpeg", "png"])


#Процес передбачення
if selected=='Simple RNN':
    model = load_model('SimpleRNN.keras')
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)
        st.image(processed_image, caption='Оброблене зображення')
        if st.button('Передбачити'):
            pred(model, processed_image)
        if st.button('Вивести графіки'):
             graph('history_SimpleRNN.json')
elif selected=='RNN with VGG16':
    model = load_model('VGG16_RNN.keras')
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = preprocess_image_vgg(image)
        st.image(processed_image, caption='Оброблене зображення')
        if st.button('Передбачити'):
            pred(model, processed_image)
        if st.button('Вивести графіки'):
             graph('history_VGG16_RNN.json')