
import io
import json
import traceback
import cv2
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os
import requests
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, TrOCRProcessor, VisionEncoderDecoderModel
from django.shortcuts import render
from tensorflow.keras.models import load_model

from ocr_web import settings
from ocr_web.settings import MEDIA_ROOT


# Create your views here.
pix2struct_processor = Pix2StructProcessor.from_pretrained(
    "google/pix2struct-docvqa-base")
pix2struct_model = Pix2StructForConditionalGeneration.from_pretrained(
    "google/pix2struct-docvqa-base")


def index(request):
    return render(request, 'index.html')


# Load TrOCR model and processor once
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten")

# load ocr_model.h5 check if the file exists

ocr_model = load_model(MEDIA_ROOT + '/mnist-model.h5')
# print if the model is loaded successfully
if ocr_model:
    print("OCR model loaded successfully.")
else:
    print("Error loading OCR model.")


def preprocess_for_doctr(pil_img, target_size=(1024, 256)):
    print("[Preprocess] Converting to RGB and resizing image...")

    pil_img = pil_img.convert("RGB")

    # Resize while maintaining aspect ratio
    orig_w, orig_h = pil_img.size
    target_w, target_h = target_size

    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    resized_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # Pad to final target size
    padded_img = Image.new("RGB", target_size, (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded_img.paste(resized_img, (paste_x, paste_y))

    print(f"[Preprocess] Final image size: {padded_img.size}")
    return padded_img


@csrf_exempt
def process_image(request):
    if request.method != 'POST' or 'image' not in request.FILES:
        return JsonResponse({'error': 'Invalid request.'}, status=400)

    image_file = request.FILES['image']

    try:
        original_img = Image.open(image_file).convert('RGB')
    except Exception as e:
        return JsonResponse({'error': f'Failed to load image: {str(e)}'}, status=400)

    results = []
    algorithms = request.POST.get('algorithms', '').lower().split(',')

    # --- CNN Prediction ---
    if 'cnn' in algorithms:
        try:
            img_cv = np.array(original_img.convert('L'))
            blurred = cv2.GaussianBlur(img_cv, (5, 5), sigmaX=1)
            resized = cv2.resize(blurred, (28, 28))
            normalized = resized.astype('float32') / 255.0
            cnn_array = np.expand_dims(normalized, axis=(0, -1))  # (1,28,28,1)

            prediction = ocr_model.predict(cnn_array)
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))

            results.append({
                'algorithm': 'CNN Model',
                'result': f'Class: {predicted_class}, Confidence: {confidence:.4f}'
            })
        except Exception as e:
            results.append({
                'algorithm': 'CNN Model',
                'result': f'Error: {str(e)}'
            })

    # --- Shared Preprocessing for All OCRs ---
    try:
        img_cv = np.array(original_img.convert('L'))
        blurred = cv2.GaussianBlur(img_cv, (5, 5), sigmaX=1)
        thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        processed_img = Image.fromarray(
            cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB))
    except Exception as e:
        return JsonResponse({'error': f'Preprocessing error: {str(e)}'}, status=500)

    # --- Tesseract OCR ---
    if 'tesseractocr' in algorithms:
        try:
            import pytesseract
            text = pytesseract.image_to_string(
                processed_img).strip() or 'No Text Detected'
            results.append({'algorithm': 'Tesseract OCR', 'result': text})
        except Exception as e:
            results.append({'algorithm': 'Tesseract OCR',
                           'result': f'Error: {str(e)}'})

    # --- PaddleOCR ---
    if 'paddleocr' in algorithms:
        try:
            from paddleocr import PaddleOCR

            # or 'en', 'ch', etc.
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            # Convert RGB to BGR for OpenCV
            img_np = np.array(original_img)[:, :, ::-1]
            result = ocr.predict(img_np)

            extracted_text = []
            for line in result:
                for word_info in line:
                    # word_info format: [coords, (text, confidence)]
                    text = word_info[1][0]
                    extracted_text.append(text)

            joined_text = "\n".join(
                extracted_text).strip() or 'No Text Detected'
            results.append({'algorithm': 'PaddleOCR', 'result': joined_text})
        except Exception as e:
            results.append({'algorithm': 'PaddleOCR',
                            'result': f'Error: {str(e)}'})

    if 'pix2struct' in algorithms:
        try:
            prompt = "Extract all text from this document"
            inputs = pix2struct_processor(
                images=original_img, text=prompt, return_tensors="pt")
            outputs = pix2struct_model.generate(**inputs)
            result_text = pix2struct_processor.batch_decode(
                outputs, skip_special_tokens=True)[0].strip()

            results.append({
                'algorithm': 'Pix2Struct',
                'result': result_text or 'No Text Detected'
            })
        except Exception as e:
            print(f"[Pix2Struct Error] {str(e)}")
            traceback.print_exc()
            results.append({
                'algorithm': 'Pix2Struct',
                'result': f'Error: {str(e)}'
            })
    # --- EasyOCR ---
    if 'easyocr' in algorithms:
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            text = reader.readtext(np.array(processed_img), detail=0)
            joined = ' '.join(text).strip() or 'No Text Detected'
            results.append({'algorithm': 'EasyOCR', 'result': joined})
        except Exception as e:
            results.append(
                {'algorithm': 'EasyOCR', 'result': f'Error: {str(e)}'})

    # --- Transformer OCR ---
    if 'transformerocr' in algorithms:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-handwritten")
            model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-handwritten")

            pixel_values = processor(
                images=processed_img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            decoded = processor.batch_decode(
                generated_ids, skip_special_tokens=True)[0].strip()

            results.append({'algorithm': 'Transformer OCR',
                           'result': decoded or 'No Text Detected'})
        except Exception as e:
            results.append({'algorithm': 'Transformer OCR',
                           'result': f'Error: {str(e)}'})

    # --- Doctr OCR ---
    if 'doctr' in algorithms:
        try:
            from doctr.models import ocr_predictor
            from doctr.io import DocumentFile
        #     image_bytes = io.BytesIO()
        #     processed_img.save(image_bytes, format='PNG')
        #     image_bytes = image_bytes.getvalue()  # ← convert BytesIO to raw bytes
        # # Create doctr document
        #     doc = DocumentFile.from_images([image_bytes])
            preprocessed_pil_img = preprocess_for_doctr(original_img)
            print(
                f"[Doctr] Step 3: Checking type of preprocessed image: {type(preprocessed_pil_img)}")
            # print size of image
            print(f"[Doctr] Step 3: Image size: {preprocessed_pil_img.size}")

            print(
                f"[Doctr] Step 3: Image size: {preprocessed_pil_img.size}, mode: {preprocessed_pil_img.mode}")

            print("[Doctr] Step 4: Converting PIL to bytes...")
            image_bytes = io.BytesIO()
            preprocessed_pil_img.save(image_bytes, format='PNG')
            image_bytes = image_bytes.getvalue()  # Convert BytesIO → raw bytes

            print("[Doctr] Step 5: Creating DocumentFile from bytes...")
            doc = DocumentFile.from_images([image_bytes])
            model = ocr_predictor(
                det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
            result = model(doc)

            extracted_text = "\n".join(
                " ".join(word.value for word in line.words)
                for page in result.pages
                for block in page.blocks
                for line in block.lines
            )

            results.append({'algorithm': 'Doctr', 'result': extracted_text})
        except Exception as e:
            results.append(
                {'algorithm': 'Doctr', 'result': f'Error: {str(e)}'})

    if results:
        return JsonResponse({'results': results})
    return JsonResponse({'error': 'No results returned.'}, status=500)


DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_API_KEY = settings.OPENROUTER_API_KEY


def summarize_result(request):
    if request.method == 'POST':
        # Get the OCR result text from the POST data
        data = json.loads(request.body)
        result_text = data.get('result_text')

        if not result_text:
            return JsonResponse({'error': 'No text provided for summarization.'}, status=400)

        try:
            # Prepare the payload for the DeepSeek API request
            payload = {
                # Correct OpenRouter DeepSeek model
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful summarizer."},
                    {"role": "user",
                        "content": f"Summarize the following text: {result_text}"}
                ],
                "temperature": 0.7,
                "max_tokens": 100  # Optional, OpenRouter supports this
            }

            headers = {
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'http://127.0.0.1:8001',
                'X-Title': 'DeepSeek Summarization',
                'X-Description': 'Summarization of OCR results using DeepSeek API',
            }

            # Send the POST request to DeepSeek API for summarization
            response = requests.post(
                DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))
            print(
                f"[DeepSeek] API Response: {response.status_code}, {response.text}")

            # Check if the request was successful
            if response.status_code == 200:
                # Update to reflect correct response format
                summary = response.json()['choices'][0]['message']['content']
                print(f"[DeepSeek] Summary: {summary}")
                if not summary:
                    return JsonResponse({'error': 'No summary returned from DeepSeek.'}, status=500)
                return JsonResponse({'summary': summary})

            else:
                return JsonResponse({'error': f"API error: {response.status_code}, {response.text}"}, status=500)

        except Exception as e:
            import traceback
            print("[Error] Exception Traceback:")
            traceback.print_exc()
            print(f"Error during summarization: {str(e)}")
            return JsonResponse({'error': f'Failed to summarize text: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def askAi(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question')
        result_text = data.get('result_text')

        if not question:
            return JsonResponse({'error': 'No question provided.'}, status=400)
        if not result_text:
            return JsonResponse({'error': 'No text provided for question.'}, status=400)

        try:
            # Correct chat-style payload
            payload = {
                # Use valid model (adjust based on availability)
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                    {"role": "user", "content": f"Context:\n{result_text}\n\nQuestion:\n{question}"}
                ],
                "temperature": 0.7,
                "max_tokens": 300  # You can increase max_tokens for longer answers
            }

            headers = {
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'http://127.0.0.1:8001',
                'X-Title': 'DeepSeek Q&A'
            }

            response = requests.post(
                DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))
            print(
                f"[AskAI] API Response: {response.status_code}, {response.text}")

            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content']
                return JsonResponse({'answer': answer})
            else:
                return JsonResponse({'error': f"API error: {response.status_code}, {response.text}"}, status=500)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': f'Failed to get answer: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
