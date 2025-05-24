import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
# from PIL import Image # PIL is not explicitly used in the core logic after conversion
import os
# import base64 # Not needed for Gradio image display in this way

# --- Configuration ---
MODEL_PATH = 'Fake-currency.keras' # Make sure this path is correct
SAMPLE_IMAGE_DIR = os.path.join(os.getcwd(), "Indian Currency Dataset", "samples")

# --- Load model once ---
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure '{MODEL_PATH}' exists and is a valid Keras model.")
    model = None # Set model to None if loading fails

# --- Preprocess image ---
def preprocess_image(image_np):
    # Image is already a NumPy array from Gradio's gr.Image(type="numpy")
    # It's usually BGR by default from OpenCV/Gradio.
    # If your model expects RGB, you might need:
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_np, (224, 224))
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)

# --- Predict ---
def predict_currency_confidence(image_np):
    if model is None:
        raise gr.Error("Model not loaded. Cannot perform prediction.")
    processed_image = preprocess_image(image_np)
    prediction = model.predict(processed_image)
    return prediction[0][0] # Confidence for the 'fake' class (assuming binary classification)

# --- Main Gradio processing function ---
def analyze_currency(image_input):
    if image_input is None:
        return (
            None, # No image to display back
            "⚠️ Please upload an image or select a sample.",
            "", # Confidence
            "", # Suggestion
            "", # Badge
            ""  # Probability distribution
        )

    try:
        confidence_fake = predict_currency_confidence(image_input) # This is P(Fake)
    except Exception as e:
        # return None, f"Error during prediction: {str(e)}", "", "", "", ""
        raise gr.Error(f"Error during prediction: {str(e)}")


    fake_percent = confidence_fake * 100
    real_percent = 100 - fake_percent

    result_md = ""
    suggestion_text = ""
    badge_html = ""

    if confidence_fake > 0.5: # Threshold for being fake
        result_md = "### 🟥 **Fake Currency Detected!**"
        suggestion_text = "🔎 Suggestion: Double-check watermark, Security Thread, and Color Pattern."
        badge_html = f"""
            <div style='margin-top:15px; text-align:center;'>
                <span style='background-color:#d9534f;color:white;padding:8px 15px;border-radius:8px;font-size:18px;font-weight:bold;'>
                    Fake Note Detected
                </span>
                <p style='font-size:16px; margin-top:10px;'>Confidence: <strong>{fake_percent:.2f}% Fake</strong></p>
            </div>
        """
    else:
        result_md = "### 🟩 **Real Currency Detected!**"
        suggestion_text = "✅ Looks authentic, but always verify manually for large transactions."
        badge_html = f"""
            <div style='margin-top:15px; text-align:center;'>
                <span style='background-color:#5cb85c;color:white;padding:8px 15px;border-radius:8px;font-size:18px;font-weight:bold;'>
                    Verified as Real
                </span>
                <p style='font-size:16px; margin-top:10px;'>Confidence: <strong>{real_percent:.2f}% Real</strong></p>
            </div>
        """

    prob_dist_text = (
        f"🧾 **Probability Distribution:**\n"
        f"   - Fake: {fake_percent:.2f}%\n"
        f"   - Real: {real_percent:.2f}%"
    )

    # Return the original image to display it, then other results
    return image_input, result_md, suggestion_text, badge_html, prob_dist_text

# --- UI Setup with Gradio Blocks ---
app_title = "💵 Fake Currency Detection App"
app_description = "Upload a Currency Image or Try Sample Images to Check Authenticity."

# Prepare sample images
sample_images_list = []
if os.path.exists(SAMPLE_IMAGE_DIR):
    for f_name in os.listdir(SAMPLE_IMAGE_DIR):
        if f_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            sample_images_list.append(os.path.join(SAMPLE_IMAGE_DIR, f_name))
else:
    print(f"Warning: Sample directory '{SAMPLE_IMAGE_DIR}' not found. No samples will be loaded.")


with gr.Blocks(theme=gr.themes.Soft(), title="Fake Currency Detection") as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{app_title}</h1>")
    gr.Markdown(f"<p style='text-align: center;'>{app_description}</p>")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Currency Image or Select Sample", height=400)
            if sample_images_list: # Only show examples if directory and images exist
                 gr.Examples(
                    examples=sample_images_list,
                    inputs=image_input,
                    label="🧪 Try Sample Images",
                    # examples_per_page=5 # Optional
                )
            else:
                gr.Markdown("_(No sample images found or directory missing)_")

            analyze_button = gr.Button("🚀 Detect Currency", variant="primary")

        with gr.Column(scale=1):
            output_image_display = gr.Image(label="Processed Image", height=400, interactive=False)
            result_markdown = gr.Markdown(label="Detection Result")
            badge_output_html = gr.HTML(label="Status")
            suggestion_output_text = gr.Textbox(label="Suggestions & Confidence", lines=3, interactive=False)
            prob_dist_output_text = gr.Markdown(label="Probability Details")


    # Wire components
    analyze_button.click(
        fn=analyze_currency,
        inputs=[image_input],
        outputs=[
            output_image_display,
            result_markdown,
            suggestion_output_text, # Swapped order with badge for better flow
            badge_output_html,
            prob_dist_output_text
        ]
    )
    
    # If an image is uploaded or a sample is clicked, also show it in the output_image_display area
    # This provides immediate feedback before clicking "Detect"
    def display_uploaded_image(img):
        if img is None:
            return None # Clear the display if no image
        return img

    image_input.change(
        fn=display_uploaded_image,
        inputs=image_input,
        outputs=output_image_display
    )


if __name__ == "__main__":
    if model is None:
        print("Cannot launch Gradio app because the model failed to load.")
    else:
        print("Launching Gradio app...")
        demo.launch()