import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shap
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from alibi.explainers import CounterfactualProto
from alibi.utils.mapping import ohe_to_ord

st.title("Lung X-Ray Diagnosis with Explainability")

# Load a pre-trained DenseNet121 model
try:
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # Binary classification
    model = Model(inputs=base_model.input, outputs=x)
    st.write("Pre-trained DenseNet121 model loaded successfully.")
except Exception as e:
    st.error(f"Error loading pre-trained model: {e}")

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image_resized = cv2.resize(image, target_size)
    if len(image_resized.shape) == 2:  # Convert grayscale to RGB
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    image_array = np.array(image_resized, dtype=np.float32) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)

# Grad-CAM function
def compute_gradcam(model, image, pred_index=None):
    try:
        last_conv_layer_name = "conv5_block16_concat"
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        st.error(f"Grad-CAM Error: {e}")
        return None

# SHAP function
def compute_shap(model, sample):
    try:
        explainer = shap.GradientExplainer(model, sample)
        shap_values = explainer.shap_values(sample)
        return shap_values
    except Exception as e:
        st.error(f"SHAP Error: {e}")
        return None

# LIME explanation function
def compute_lime_explanation(model, image, target_size=(224, 224)):
    def predict_fn(images):
        preprocessed_images = np.array([preprocess_input(cv2.resize(img, target_size)) for img in images])
        return model.predict(preprocessed_images)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image.astype('double'),
        predict_fn,
        top_labels=1,  # Only explain the top label
        hide_color=0,
        num_samples=50  # Reduced samples for faster explanation
    )
    return explanation

# Alibi Counterfactual explanation function
def compute_counterfactual(model, instance, target_shape=(224, 224, 3)):
    try:
        predictor = lambda x: model.predict(x)
        cf = CounterFactualProto(
            predictor=predictor,
            shape=target_shape,
            use_kdtree=True,
            theta=1.0,
            max_iterations=1000,
            feature_range=(0, 1)
        )
        cf.fit(
            instance,
            eps=0.01
        )
        explanation = cf.explain(instance)
        return explanation
    except Exception as e:
        st.error(f"Counterfactual Error: {e}")
        return None

# File uploader and image handling
uploaded_file = st.file_uploader("Upload a Lung X-Ray Image", type=["jpg", "png"])
if uploaded_file:
    try:
        image = plt.imread(uploaded_file)
        st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
        preprocessed_image = preprocess_image(image)

        # Predict the class
        prediction = model.predict(preprocessed_image)
        predicted_class = "Diseased" if prediction[0] > 0.5 else "Healthy"
        confidence = prediction[0][0] * 100 if predicted_class == "Diseased" else (1 - prediction[0][0]) * 100
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Grad-CAM
        heatmap = compute_gradcam(model, preprocessed_image)
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            superimposed_image = heatmap_colored * 0.4 + image / 255.0
            superimposed_image = np.clip(superimposed_image, 0.0, 1.0)

            st.subheader("Grad-CAM Explanation")
            st.image(superimposed_image, caption="Grad-CAM Heatmap", use_container_width=True)

        # SHAP Explanation with alternative rendering
        st.subheader("SHAP Explanation")
        shap_values = compute_shap(model, preprocessed_image)
        if shap_values is not None:
            st.write("Visualizing SHAP values for the uploaded image:")
            shap.image_plot(shap_values, preprocessed_image, show=False)
            plt.savefig("shap_explanation.png")  # Save the plot to a file
            st.image("shap_explanation.png", caption="SHAP Explanation", use_container_width=True)

        # LIME Explanation
        st.subheader("LIME Explanation")
        lime_explanation = compute_lime_explanation(model, image)
        temp, mask = lime_explanation.get_image_and_mask(
            label=lime_explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        lime_image = mark_boundaries(temp / 255.0, mask)

        st.image(lime_image, caption="LIME Explanation", use_container_width=True)

        # Alibi Counterfactual Explanation
        st.subheader("Counterfactual Explanation")
        counterfactual_explanation = compute_counterfactual(model, preprocessed_image[0])
        if counterfactual_explanation is not None:
            st.write("Counterfactual Explanation generated:")
            st.json(counterfactual_explanation)

    except Exception as e:
        st.error(f"Error during prediction or explanation: {e}")
