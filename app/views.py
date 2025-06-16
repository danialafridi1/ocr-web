
import io
import json
import cv2
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from django.shortcuts import render
from tensorflow.keras.models import load_model

from ocr_web import settings
from ocr_web.settings import MEDIA_ROOT


# Create your views here.


def index(request):
    return render(request, 'index.html')



                

# Load TrOCR model and processor once
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten")

# load ocr_model.h5 check if the file exists

ocr_model = load_model(MEDIA_ROOT+ '/mnist-model.h5');
# print if the model is loaded successfully
if ocr_model:
    print("OCR model loaded successfully.")
else:
    print("Error loading OCR model.");


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


DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/completions"  # DeepSeek API endpoint
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
                "model": "deepseek/deepseek-r1:free",  # The DeepSeek R1 free model
                # Use "prompt" instead of "input"
                "prompt": f"Summarize the following text: {result_text}",
                "parameters": {
                    "temperature": 0.7,  # Adjust as needed
                    # Limit summary length (adjust as needed)
                    "max_tokens": 100,
                }
            }

            headers = {
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            }

            # Send the POST request to DeepSeek API for summarization
            response = request.post(
                DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))

            # Check if the request was successful
            if response.status_code == 200:
                summary = response.json().get('choices')[0].get(
                    'text')  # Update to reflect correct response format
                return JsonResponse({'summary': summary})
            else:
                return JsonResponse({'error': f"API error: {response.status_code}, {response.text}"}, status=500)

        except Exception as e:
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

        # here you have to send promot as question promot related to context result_text
        # and then send the response of the model back to the client
        try:
            # Prepare the payload for the DeepSeek API request
            payload = {
                "model": "deepseek/deepseek-r1:free",  # The DeepSeek
                # Use "prompt"
                "prompt": f"Answer the following question based on the context: \n\nContext: \n{result_text}\n\nQuestion: \n{question}",
                "parameters": {
                    "temperature": 0.7,  # Adjust as needed
                    # Limit response length (adjust as needed)
                    "max_tokens": 100,
                }
            }

            headers = {
                'Authorization': f'bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            }
            response = requests.post(
                DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                summary = response.json().get('choices')[0].get(
                    'text')  # Update to reflect correct response format
                return JsonResponse({'answer': summary})
            else:
                return JsonResponse({'error': f"API error: {response.status_code}, {response.text}"}, status=500)
        except Exception as e:
            return JsonResponse({'error': f'Failed to summarize text: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
