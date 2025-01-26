import os
import io
import torch
import concurrent.futures
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
import easyocr
from paddleocr import PaddleOCR
import openai
import base64
import asyncio
import tempfile
from datetime import datetime, timedelta
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

# Environment variable to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set up FastAPI
app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins like ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


print("\nRegistered Routes:")
for route in app.routes:
    if isinstance(route, APIRoute):
        print(f"Path: {route.path}, Methods: {route.methods}")



current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "../frontend")

# Serve static files at `/static`
app.mount("/static", StaticFiles(directory=frontend_dir, html=True), name="static")

# Serve the main frontend file
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model Quantization Configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,                      # Enable 8-bit quantization
    llm_int8_enable_fp32_cpu_offload=True,  # Offload FP32 layers to CPU if GPU memory is insufficient
)

# Pre-caching Models
print("Loading models...")
florence_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True
).to(device)
floence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# OpenAI 
openai.api_key = ""

# Initialize OCR
reader = easyocr.Reader(['ar'], gpu=True)
ocr = PaddleOCR(use_gpu=True, lang='en')

print("Models loaded successfully!")

# Function to resize with padding while maintaining aspect ratio
def resize_with_padding(image, target_size=(256, 256), color=(0, 0, 0)):
    """
    Resize image while maintaining aspect ratio and pad to target size.
    """
    original_width, original_height = image.size
    ratio = min(target_size[0] / original_width, target_size[1] / original_height)

    # Resize with the calculated ratio
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image_resized = image.resize((new_width, new_height), Image.BICUBIC)

    # Create a blank canvas and paste the resized image
    new_image = Image.new("RGB", target_size, color)
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(image_resized, (paste_x, paste_y))

    return new_image

# Preprocessing function
def preprocess_image(image: Image.Image, size=(512, 512)):
    """Resize image while maintaining aspect ratio and pad to fixed size."""
    return resize_with_padding(image, target_size=size)

# OCR with multi-threading
def run_ocr(image):
    print("Running OCR...")
    image_resized = image.resize((800, 800), Image.LANCZOS)
    image_np = np.array(image_resized)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        arabic_future = executor.submit(reader.readtext, image_np)
        paddle_future = executor.submit(ocr.ocr, image_np)
        arabic_results = arabic_future.result()
        paddle_results = paddle_future.result()

    arabic_text = " ".join([res[1] for res in arabic_results if res[2] >= 0.55])
    if not paddle_results or len(paddle_results) == 0 or not paddle_results[0]:
        return f"{arabic_text}".strip()
    english_text = " ".join([line[1][0] for line in paddle_results[0] if line[1][1] >= 0.7])
    return f"{arabic_text} {english_text}".strip()


# Florence-2 model
def run_florence(image):
    print("Running Florence...")
    task_prompt = "Describe the content of this <image>."
    inputs = floence_processor(
        text=task_prompt, images=image, return_tensors="pt"
    ).to(device, torch_dtype)
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,
        num_beams=3,
    )
    generated_text = floence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return generated_text

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# OpenAI ChatGPT-4 Vision model
def run_openai(image_path, question, system_content):
    print("Running OpenAI ChatGPT-4 Vision...")
    
    # Getting the base64 string
    base64_image = encode_image(image_path)

    # Use OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o",  # Use the GPT-4 Vision model
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
    )

    # Extract the response text
    response_message = response.choices[0].message.content
    return response_message


def run_openai_vqa(image_path, question):
    print("Running OpenAI ChatGPT-4 Vision...")
    
    # Getting the base64 string
    base64_image = encode_image(image_path)

    # Use OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o",  # Use the GPT-4 Vision model
        messages=[
            {"role": "system", "content": "أنت مساعد ذكي مصمم لمساعدة المكفوفين أو ضعاف البصر من خلال وصف الصور التي يلتقطونها باستخدام كاميراتهم. الصور مأخوذة من محيط المستخدم المباشر، وهي لأشياء أو أماكن قريبة منهم. قدم أوصافًا دقيقة ومفصلة باستخدام لغة عربية بسيطة وسهلة الفهم. ركز على تحديد الأشياء، مواقعها بالنسبة للمستخدم، ألوانها، أشكالها، قوامها، وما إذا كانت هناك أي مخاطر محتملة أو تفاصيل تستدعي الانتباه. إذا كان السؤال غير واضح، اطلب توضيحًا بلطف وساعد المستخدم على صياغة استفساره. كن متعاطفًا وركز على جعل الأوصاف مفيدة وسهلة التخيّل."},
            {"role": "user", "content": [
                {"type": "image_url", 
                "image_url": 
                    {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    }
                },
                {"type": "text", "text": question}
            ]
            },
            
        ],
        
    )

    # Extract the response text
    response_message = response.choices[0].message.content
    return response_message
    
def run_openai_quick_caption(image_path, question):
    print("Running OpenAI ChatGPT-4 Vision...")
    
    # Getting the base64 string
    base64_image = encode_image(image_path)

    # Use OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o",  # Use the GPT-4 Vision model
        messages=[
{
  "role": "system",
  "content": "أنت مساعد ذكي مصمم لتقديم وصف سريع عن البيئة المحيطة بالمستخدم بناءً على الصور التي يلتقطها. استخدم لغة عربية بسيطة وسهلة الفهم لتحديد الأشياء الرئيسية في الصورة، مثل الأثاث، الألوان، المواد، والمسافات التقريبية بين الأشياء. حاول أن يكون الوصف موجزًا وشاملًا يركز على النقاط الأكثر أهمية أو لفتًا للانتباه."
},
            {"role": "user", "content": [
                {"type": "image_url", 
                "image_url": 
                    {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    }
                },
                {"type": "text", "text": question}
            ]
            },
            
        ],
        
    )

    # Extract the response text
    response_message = response.choices[0].message.content
    return response_message

def run_openai_locator(image_path, question):
    print("Running OpenAI ChatGPT-4 Vision...")
    
    # Getting the base64 string
    base64_image = encode_image(image_path)

    # Use OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o",  # Use the GPT-4 Vision model
        messages=[
{
  "role": "system",
  "content": "أنت مساعد ذكي متخصص في تحديد مواقع الأشخاص أو الأشياء في الصور الملتقطة بواسطة المستخدم. استخدم لغة عربية بسيطة لتحديد مواقع العناصر بالنسبة للمستخدم (مثل 'أمامك مباشرة' أو 'إلى يمينك على بعد متر واحد'). إذا لم يكن العنصر واضحًا في الصورة، استفسر بلطف عن المزيد من التفاصيل لمساعدة المستخدم بشكل أفضل."
},
            {"role": "user", "content": [
                {"type": "image_url", 
                "image_url": 
                    {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    }
                },
                {"type": "text", "text": question}
            ]
            },
            
        ],
        
    )

    # Extract the response text
    response_message = response.choices[0].message.content
    return response_message


# Helper function to delete files after a delay
async def delete_file_after_delay(file_path, delay_seconds=60):
    """Deletes the file after the specified delay."""
    await asyncio.sleep(delay_seconds)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted temporary file: {file_path}")

# Function to save image to a temporary folder
def save_temp_image(image: Image.Image):
    """Saves the image as a temporary file and returns the file path."""
    temp_dir = tempfile.gettempdir()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    temp_file_path = os.path.join(temp_dir, f"temp_image_{timestamp}.jpeg")
    image.save(temp_file_path, format="JPEG")
    print(f"Saved image to temporary file: {temp_file_path}")
    return temp_file_path

# Model router
def run_model(image, image_path, prompt):
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ["اقرا", "الكلام","كلام","نص", "النص"]):
        result = run_ocr(image)
        print("ocr")
        print(result)
        if result is None or result.strip() == "" or not result:
            return "لم يتم العثور على نص في الصورة."
        return result
        
    elif any(word in prompt_lower for word in ["وصف سريع", "وصف بسيط"]):
        print("quick caption")
        system_content = "أنت مساعد ذكي مصمم لتقديم وصف سريع عن البيئة المحيطة بالمستخدم الكفيف بناءً على الصور التي يلتقطونها. استخدم لغة عربية بسيطة وسهلة الفهم لتحديد الأشياء الرئيسية في الصورة، مثل الأثاث، الألوان، المواد، والمسافات التقريبية بين الأشياء. حاول أن يكون الوصف موجزًا وشاملًا يركز على النقاط الأكثر أهمية أو لفتًا للانتباه."        
        return run_openai(image_path, prompt, system_content)
        
    elif any(word in prompt_lower for word in ["كم", "كم عدد", "عدد"]):
        print("quantitive description")
        system_content = "أنت مساعد ذكي مصمم لتقديم إجابة سريعة عن عدد الأشياء الموجودة في الصورة عندما يسأل المستخدم الكفيف 'كم'. استخدم لغة عربية بسيطة ودقيقة لتحديد عدد العناصر المعنية بناءً على الصورة، مثل 'هناك ثلاثة كراسي' أو 'يوجد خمسة أشخاص'. إذا كانت العناصر غير واضحة أو تحتاج إلى توضيح إضافي، استفسر بلطف لمساعدة المستخدم بشكل أفضل."
        return run_openai(image_path, prompt, system_content)
    

    elif any(word in prompt_lower for word in ["بكم", "سعر", "كم سعر", "سعره", "سعرها"]):
        print("price caption")
        system_content = "أنت مساعد ذكي مصمم لتقديم إجابة سريعة عن سعر الأشياء عندما يسأل المستخدم الكفيف 'بكم'. إذا كانت الصورة تحتوي على معلومات عن الأسعار، قدم الإجابة بدقة مثل 'سعره 20 ريالاً'. وإذا لم تكن المعلومات متوفرة، وضح ذلك بلطف للمستخدم واقترح طرقًا للحصول على التفاصيل المطلوبة."
        return run_openai(image_path, prompt, system_content)
    
    elif any(word in prompt_lower for word in ["أين", "وين"]):
        print("locator")
        system_content = "أنت مساعد ذكي متخصص في تحديد مواقع الأشخاص أو الأشياء في الصور الملتقطة بواسطة المستخدم الكفيف. استخدم لغة عربية بسيطة لتحديد مواقع العناصر بالنسبة للمستخدم (مثل 'أمامك مباشرة' أو 'إلى يمينك على بعد متر واحد'). إذا لم يكن العنصر واضحًا في الصورة، استفسر بلطف عن المزيد من التفاصيل لمساعدة المستخدم بشكل أفضل."
        return run_openai(image_path, prompt, system_content)
    else:
        print("vqa")
        system_content = "أنت مساعد ذكي مصمم لمساعدة المكفوفين أو ضعاف البصر من خلال وصف الصور التي يلتقطونها باستخدام كاميراتهم. الصور مأخوذة من محيط المستخدم المباشر، وهي لأشياء أو أماكن قريبة منهم. قدم أوصافًا دقيقة ومفصلة باستخدام لغة عربية بسيطة وسهلة الفهم. ركز على تحديد الأشياء، مواقعها بالنسبة للمستخدم، ألوانها، أشكالها، قوامها، وما إذا كانت هناك أي مخاطر محتملة أو تفاصيل تستدعي الانتباه. إذا كان السؤال غير واضح، اطلب توضيحًا بلطف وساعد المستخدم على صياغة استفساره. كن متعاطفًا وركز على جعل الأوصاف مفيدة وسهلة التخيّل."
        return run_openai(image_path, prompt, system_content)

# API Endpoint
@app.post("/process")
async def process_frame(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        print("Processing image...")
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Preprocess image
        image = preprocess_image(image)
        print("Image preprocessed successfully.")

        # Save the image to a temporary folder
        temp_file_path = save_temp_image(image)
        print(temp_file_path)

        # Schedule deletion of the temporary file
        asyncio.create_task(delete_file_after_delay(temp_file_path))

        # Run model based on prompt
        result = run_model(image, temp_file_path, prompt)
        
        print(f"Result: {result}")
        return JSONResponse(content={"result": result}, status_code=200)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/")
def health_check():
    return {"status": "Running", "message": "The API is ready to process images!"}

# Run Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
