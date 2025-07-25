# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
import pytesseract
import io
import os
import base64
from dotenv import load_dotenv
import google.generativeai as genai
from docx import Document
from datetime import timedelta, datetime # Import datetime for type checking

# Load environment variables from .env file
load_dotenv()

# --- Tesseract OCR Configuration (CRUCIAL for Windows) ---
# If you are on Windows, you MUST specify the path to your tesseract.exe
# If Tesseract-OCR is not installed or the path is incorrect, OCR will fail.
# Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# UNCOMMENT THE LINE BELOW AND UPDATE THE PATH IF NECESSARY FOR YOUR SYSTEM:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # <--- UPDATE THIS PATH!

# For PDF OCR: pytesseract relies on Poppler utilities for PDF processing.
# If you intend to OCR PDFs, you need to:
# 1. Download Poppler for Windows (e.g., from https://github.com/oschwartz10612/poppler-windows/releases)
# 2. Extract it and add its 'bin' directory to your system's PATH environment variable.
# Without Poppler, PDF OCR will likely result in an error or empty text.


# --- Firebase Admin SDK Initialization ---
serviceAccountKeyPath = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY_PATH', 'serviceAccountKey.json')
firebase_storage_bucket_name = os.getenv('FIREBASE_STORAGE_BUCKET')

# --- DEBUGGING PRINTS ---
print(f"DEBUG: serviceAccountKeyPath from .env: {serviceAccountKeyPath}")
print(f"DEBUG: FIREBASE_STORAGE_BUCKET from .env: {firebase_storage_bucket_name}")
# --- END DEBUGGING PRINTS ---

if not serviceAccountKeyPath or not os.path.exists(serviceAccountKeyPath):
    raise FileNotFoundError(f"Firebase service account key file not found at: {serviceAccountKeyPath}. Please check FIREBASE_SERVICE_ACCOUNT_KEY_PATH in your .env file.")
if not firebase_storage_bucket_name:
    raise ValueError("FIREBASE_STORAGE_BUCKET not found in environment variables. Please set it in your .env file.")

cred = credentials.Certificate(serviceAccountKeyPath)
firebase_admin.initialize_app(cred, {
    'storageBucket': firebase_storage_bucket_name # Use the name read from .env
})
db = firestore.client()

# Explicitly get the bucket using the name, not just the default
bucket = storage.bucket(name=firebase_storage_bucket_name) 

# --- DEBUGGING PRINTS ---
print(f"DEBUG: Successfully initialized Firebase app with storageBucket: {firebase_storage_bucket_name}")
print(f"DEBUG: Attempting to access bucket: {bucket.name}")
# --- END DEBUGGING PRINTS ---


# --- Gemini API Initialization ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

app = Flask(__name__)
CORS(app)

# --- Helper Functions ---

def extract_text_from_file(file_bytes, file_mimetype, filename):
    """
    Extracts text from various file types.
    Returns extracted_text.
    """
    ocr_text = ''

    if file_mimetype.startswith('image/'):
        try:
            img = Image.open(io.BytesIO(file_bytes))
            ocr_text = pytesseract.image_to_string(img)
            print(f"OCR successful for image: {filename}")
        except Exception as e:
            print(f"Error processing image '{filename}' for OCR: {e}")
            ocr_text = f"OCR failed for image '{filename}'. Error: {e}"

    elif file_mimetype == 'application/pdf':
        print(f"Attempting OCR for PDF: {filename}. Requires Tesseract and Poppler.")
        try:
            ocr_text = pytesseract.image_to_string(io.BytesIO(file_bytes))
            print(f"OCR successful for PDF: {filename}")
        except Exception as e:
            print(f"Error during PDF OCR for '{filename}' (check Poppler installation/PATH): {e}")
            ocr_text = f"OCR for PDF '{filename}' failed. Ensure Tesseract and Poppler are correctly installed and configured on the backend server for PDF processing. Error: {e}"

    elif file_mimetype == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': # .docx
        print(f"Attempting text extraction for DOCX: {filename}")
        try:
            doc = Document(io.BytesIO(file_bytes))
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            ocr_text = "\n".join(full_text)
            print(f"Text extraction successful for DOCX: {filename}")
        except Exception as e:
            print(f"Error extracting text from DOCX '{filename}': {e}")
            ocr_text = f"Text extraction for DOCX '{filename}' failed. Error: {e}"

    elif file_mimetype == 'application/msword': # .doc (older format)
        print(f"Attempting text extraction for DOC: {filename}")
        ocr_text = f"Direct text extraction for older .doc files is not supported by this backend. Please convert '{filename}' to .docx or PDF for processing."
    else:
        ocr_text = f"Text extraction not supported for file type: {file_mimetype}. No digital copy extracted."
    
    # If text extraction failed or resulted in empty string, provide a generic message
    if not ocr_text.strip():
        ocr_text = f"No readable text found in '{filename}' or text extraction failed. File type: {file_mimetype}."

    return ocr_text

def is_medical_file_ai(text_content):
    """Uses Gemini API to classify if text is medical. Returns True/False."""
    if not text_content or len(text_content.strip()) < 50 or "could not extract text" in text_content.lower() or "ocr failed" in text_content.lower():
        print("OCR text too short or indicates failure, classifying as non-medical for AI check.")
        return False

    prompt = f"Is the following document content related to medical records, patient health, or clinical notes? Respond with only 'YES' or 'NO'.\n\nDocument Content:\n{text_content[:2000]}"

    try:
        response = model.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            classification_answer = response.candidates[0].content.parts[0].text.strip().upper()
            print(f"AI Classification Raw Response: '{classification_answer}'")
            return 'YES' in classification_answer
        print("AI Classification: No valid candidate found in response.")
        return False
    except Exception as e:
        print(f"Error during AI classification API call: {e}")
        return False

def get_summary_from_ai(text_content):
    """Uses Gemini API to summarize text."""
    if not text_content or len(text_content.strip()) < 50:
        return "Not enough content to generate a meaningful summary."
    
    prompt = f"Please summarize the following medical record text concisely, highlighting key information such as patient details, diagnoses, medications, and follow-up plans. Keep it under 150 words.\n\nMedical Record:\n{text_content[:2000]}"
    try:
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"Error during summarization AI call: {e}")
        return f"Failed to generate summary: {e}"

def get_analysis_from_ai(text_content):
    """Uses Gemini API to analyze text."""
    if not text_content or len(text_content.strip()) < 50:
        return "Not enough content to generate a meaningful analysis."

    analysis_prompt = f"Analyze the following medical record text and extract key details such as patient name, date of birth, main diagnosis, prescribed medications (with dosage if available), and any follow-up instructions. Present this information clearly and concisely. If any information is missing or unclear, state that. Keep the response to a maximum of 250 words.\n\nMedical Record:\n{text_content[:2500]}"
    try:
        response = model.generate_content(analysis_prompt)
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"Error during analysis AI call: {e}")
        return f"Failed to analyze document: {e}"


# --- API Endpoints ---

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'medicalFile' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['medicalFile']
    user_id = request.form.get('userId')
    category = request.form.get('category')
    # confirm_non_medical is no longer used to block upload, but can be kept if needed for other logic
    # confirm_non_medical = request.form.get('confirmNonMedical') == 'true'

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not user_id or not category:
        return jsonify({'error': 'User ID and category are required'}), 400

    file_bytes = file.read()
    file_mimetype = file.mimetype.lower()
    filename = file.filename
    
    # Generate a unique filename for Cloud Storage to avoid conflicts
    unique_filename = f"{user_id}/{os.urandom(16).hex()}_{filename}"

    try:
        # 1. Upload original file to Firebase Cloud Storage
        blob = bucket.blob(unique_filename)
        blob.upload_from_string(file_bytes, content_type=file_mimetype)
        print(f"File uploaded to Cloud Storage: {unique_filename}")

        # 2. Extract text for OCR and AI processing
        ocr_text = extract_text_from_file(file_bytes, file_mimetype, filename)
        
        # 3. AI Classification (still performed and stored, but doesn't block upload)
        is_medical = is_medical_file_ai(ocr_text)
        print(f"AI classified as medical: {is_medical}")

        # Removed: if not is_medical and not confirm_non_medical: return 202

        # 4. Automatic AI Analysis and Summarization
        analysis_result = get_analysis_from_ai(ocr_text)
        summary_result = get_summary_from_ai(ocr_text)

        # Store metadata (including Storage path) in Firestore
        doc_ref = db.collection('records').document()
        record_data = {
            'id': doc_ref.id,
            'userId': user_id,
            'name': filename,
            'type': file_mimetype,
            'size': file.content_length,
            'uploadDate': firestore.SERVER_TIMESTAMP,
            'category': category,
            'ocrText': ocr_text,
            'isMedical': is_medical,
            'storagePath': unique_filename, # Store the Cloud Storage path
            'analysisResult': analysis_result,
            'summaryResult': summary_result,
            'uploadedAt': firestore.SERVER_TIMESTAMP
        }
        doc_ref.set(record_data)

        # Convert Firestore Timestamp objects for response
        record_data['uploadDate'] = record_data['uploadDate'].isoformat() if isinstance(record_data['uploadDate'], datetime) else str(record_data['uploadDate'])
        record_data['uploadedAt'] = record_data['uploadedAt'].isoformat() if isinstance(record_data['uploadedAt'], datetime) else str(record_data['uploadedAt'])


        return jsonify({'message': 'File uploaded and processed successfully', 'record': record_data}), 201

    except Exception as e:
        print(f"Unhandled error during file upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/records', methods=['GET'])
def get_records():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    try:
        records_ref = db.collection('records').where('userId', '==', user_id).stream()
        records = []
        for doc in records_ref:
            record_data = doc.to_dict()
            # Generate signed URL for download/preview
            if 'storagePath' in record_data:
                blob = bucket.blob(record_data['storagePath'])
                # URL valid for 1 hour
                record_data['downloadUrl'] = blob.generate_signed_url(timedelta(hours=1), method='GET')
            else:
                record_data['downloadUrl'] = None # No storage path found

            # Convert Firestore Timestamp (which are datetime objects) to ISO string for frontend compatibility
            if 'uploadDate' in record_data and isinstance(record_data['uploadDate'], datetime):
                record_data['uploadDate'] = record_data['uploadDate'].isoformat()
            if 'uploadedAt' in record_data and isinstance(record_data['uploadedAt'], datetime):
                record_data['uploadedAt'] = record_data['uploadedAt'].isoformat()
            records.append(record_data)
        return jsonify(records), 200
    except Exception as e:
        print(f"Error fetching records: {e}")
        return jsonify({'error': 'Failed to fetch records'}), 500

@app.route('/api/records/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    try:
        record_ref = db.collection('records').document(record_id)
        record_doc = record_ref.get()

        if not record_doc.exists:
            return jsonify({'error': 'Record not found'}), 404
        
        doc_data = record_doc.to_dict()
        if doc_data.get('userId') != user_id:
            return jsonify({'error': 'Unauthorized to delete this record'}), 403

        # Delete file from Cloud Storage first
        if 'storagePath' in doc_data and doc_data['storagePath']:
            try:
                blob = bucket.blob(doc_data['storagePath'])
                blob.delete()
                print(f"Deleted file from Cloud Storage: {doc_data['storagePath']}")
            except Exception as storage_e:
                print(f"Warning: Could not delete file from Cloud Storage ({doc_data['storagePath']}): {storage_e}")
                # Continue to delete Firestore record even if Storage deletion fails

        record_ref.delete()
        return jsonify({'message': 'Record deleted successfully'}), 200
    except Exception as e:
        print(f"Error deleting record: {e}")
        return jsonify({'error': 'Failed to delete record'}), 500

# These endpoints are now primarily for internal use or direct backend testing.
# Frontend will use pre-computed results from Firestore.
@app.route('/api/summarize', methods=['POST'])
def summarize_text_api():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Text is required for summarization'}), 400
    return jsonify({'summary': get_summary_from_ai(text)}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_document_api():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Text is required for analysis'}), 400
    return jsonify({'analysis': get_analysis_from_ai(text)}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)



