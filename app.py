import gradio as gr
from openai import OpenAI
import os
import tempfile
try:
    import pyperclip
except ImportError:
    pyperclip = None

# LM Studio Configuration
# Ensure LM Studio server is running on this URL
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"

def get_client(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)

def save_translation_to_file(text):
    if not text:
        return gr.update(visible=False)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp:
        tmp.write(text)
        return gr.update(value=tmp.name, visible=True)

def get_clipboard_text():
    if pyperclip:
        try:
            return pyperclip.paste()
        except Exception as e:
            return f"Error pasting: {str(e)}"
    else:
        return "Error: pyperclip module not installed."

def create_formatted_prompt(source_lang, target_lang, text):
    code_map = {
        "English": "en",
        "Korean": "ko",
        "Japanese": "ja",
        "Chinese": "zh",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Russian": "ru"
    }
    
    source_code = code_map.get(source_lang, "auto")
    target_code = code_map.get(target_lang, "en") 

    if source_lang == "Auto Detect":
        return (
            f"You are a professional translator. "
            f"Identify the language of the following text and translate it into {target_lang} ({target_code}). "
            f"Your goal is to accurately convey the meaning and nuances of the original text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {target_lang} translation, without any additional explanations or commentary. Please translate the following text into {target_lang}:\n\n"
            f"{text}"
        )

    return (
        f"You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target_lang} translation, without any additional explanations or commentary. Please translate the following {source_lang} text into {target_lang}:\n\n\n"
        f"{text}"
    )

def split_text_into_chunks(text, max_chunk_size=1500):
    """
    Iterative splitter that respects separators to ensure semantic integrity.
    Strategy: 
    1. Decompose text into small 'atoms' (segments <= max_chunk_size) using hierarchy of separators.
    2. Merge 'atoms' back together into chunks to maximize size without exceeding limit.
    """
    if not text:
        return []
        
    # Separators to try in order. Last fallback is character-based splitting by logic below.
    separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
    
    # Initial state: one giant segment
    segments = [text]
    
    # 1. Top-Down Decomposition
    for sep in separators:
        new_segments = []
        for seg in segments:
            if len(seg) <= max_chunk_size:
                new_segments.append(seg)
            else:
                # Split by current separator
                parts = seg.split(sep)
                
                # If split didn't actually split (only 1 part), keep it and try next sep
                if len(parts) == 1:
                    new_segments.append(seg)
                else:
                    # Re-attach separator to preserve it in the text
                    # We attach 'sep' to every part except the last one
                    processed_parts = [p + sep for p in parts[:-1]]
                    processed_parts.append(parts[-1])
                    
                    new_segments.extend(processed_parts)
        segments = new_segments
        
    # 2. Final Hard-Split Pass
    # If any segment is STILL too large (e.g. giant block of no spaces), force split it characters
    final_atoms = []
    for seg in segments:
        if len(seg) > max_chunk_size:
            for i in range(0, len(seg), max_chunk_size):
                final_atoms.append(seg[i:i+max_chunk_size])
        else:
            final_atoms.append(seg)
            
    # 3. Bottom-Up Merge
    # Group atoms into chunks
    chunks = []
    current_chunk = ""
    
    for atom in final_atoms:
        if len(current_chunk) + len(atom) <= max_chunk_size:
            current_chunk += atom
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = atom
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def translate_streaming(text, source_lang, target_lang, model_name, temperature, base_url, chunk_size=1500):
    """
    Streaming translation for the text input tab with auto-chunking.
    """
    if not text:
        yield "", ""
        return
    
    try:
        client = get_client(base_url, DEFAULT_API_KEY)
        
        # Split text into chunks
        chunks = split_text_into_chunks(text, int(chunk_size))
        total_chunks = len(chunks)
        
        full_translation = ""
        
        for i, chunk in enumerate(chunks):
            progress_str = f"Processing chunk {i+1} of {total_chunks}..."
            full_prompt = create_formatted_prompt(source_lang, target_lang, chunk)
            
            # Stream the current chunk
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=temperature,
                stream=True, 
            )
            
            chunk_accumulated = ""
            for stream_chunk in response:
                if stream_chunk.choices[0].delta.content is not None:
                    content = stream_chunk.choices[0].delta.content
                    chunk_accumulated += content
                    # Yield previously finished chunks + current partial chunk
                    yield full_translation + chunk_accumulated, progress_str
            
            # Append finished chunk to full translation with spacing
            full_translation += chunk_accumulated + "\n\n"
            yield full_translation, f"Completed {i+1} of {total_chunks} chunks."

    except Exception as e:
        yield full_translation + f"\n[Error translating chunk {i+1}: {str(e)}]\n\nPlease ensure LM Studio is running and the server is started at {base_url}.", f"Error on chunk {i+1}"

def translate_file_process(file_obj, source_lang, target_lang, model_name, temperature, base_url, chunk_size=1500, progress=gr.Progress()):
    """
    File translation logic with chunking and progress bar.
    """
    if file_obj is None:
        yield "Please upload a file first.", None
        return

    # Handle file_obj being a string (path) or a file object
    if isinstance(file_obj, str):
        file_path = file_obj
    elif hasattr(file_obj, "name"):
        file_path = file_obj.name
    else:
        yield "Error: Invalid file object received.", None
        return

    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        yield f"Error reading file: {str(e)}", None
        return

    client = get_client(base_url, DEFAULT_API_KEY)
    
    chunks = split_text_into_chunks(text, int(chunk_size))
    total_chunks = len(chunks)
    full_translation = ""
    
    yield f"Starting translation of {total_chunks} chunks...", None
    
    # Iterate through chunks with progress bar
    for i, chunk in enumerate(progress.tqdm(chunks, desc="Translating File")):
        try:
            full_prompt = create_formatted_prompt(source_lang, target_lang, chunk)
            
            # Non-streaming call for each chunk
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=temperature,
                stream=False, 
            )
            chunk_translation = response.choices[0].message.content or ""
            full_translation += chunk_translation + "\n\n"
            
            # Update preview with current progress
            yield full_translation, None
            
        except Exception as e:
            error_msg = f"\n[Error translating chunk {i+1}: {str(e)}]\n"
            full_translation += error_msg
            yield full_translation, None

    # Save to a new file
    original_filename = os.path.basename(file_path)
    name, ext = os.path.splitext(original_filename)
    output_filename = f"translated_{name}{ext}"
    
    # Save in the same directory as the temp file or a temp dir?
    output_path = os.path.join(os.path.dirname(file_path), output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_translation)
        
    yield full_translation, output_path


# Custom CSS for a better look
custom_css = """
body { font-family: 'Segoe UI', sans-serif; }
.gradio-container { max-width: 95% !important; }
"""

with gr.Blocks(css=custom_css, title="LM Studio Translator") as demo:
    gr.Markdown(
        """
        # 🤖 AI Universal Translator
        Using local LLM via LM Studio (OpenAI Compatible Endpoint)
        """
    )
    
    # Main Layout
    with gr.Row():
        # LEFT COLUMN: Settings
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Settings (설정)")
                
                model_name_input = gr.Textbox(
                    label="Model Name (모델 이름)", 
                    value="translategemma-12b-it", 
                    placeholder="translategemma-12b-it",
                    info="LM Studio에 로드된 모델 ID"
                )
                
                base_url_input = gr.Textbox(
                    label="Server URL (서버 주소)", 
                    value=DEFAULT_BASE_URL,
                    info="LM Studio 로컬 서버 주소"
                )
                
                temp_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.1, 
                    label="Temperature (창의성/정확도)", 
                    info="낮을수록 더 정확합니다"
                )
                
                gr.Markdown("---")
                
                from_lang_drop = gr.Dropdown(
                    choices=["Auto Detect", "English", "Korean", "Japanese", "Chinese", "Spanish", "French", "German", "Russian"], 
                    value="Auto Detect", 
                    label="Source Language (출발 언어)"
                )
                
                to_lang_drop = gr.Dropdown(
                    choices=["English", "Korean", "Japanese", "Chinese", "Spanish", "French", "German", "Russian"], 
                    value="Korean", 
                    label="Target Language (도착 언어)"
                )

        # RIGHT COLUMN: Translation Area
        with gr.Column(scale=3):
            with gr.Tabs():
                
                # TAB 1: Text Translation
                with gr.TabItem("📝 Text Translation (텍스트 번역)"):
                    with gr.Row():
                        input_text = gr.Textbox(
                            label="Input Text (입력 텍스트)", 
                            placeholder="번역할 내용을 입력하세요... (Enter text to translate...)", 
                            elem_id="input_text"
                        )
                        output_text = gr.Textbox(
                            label="Translation (번역 결과)", 
                            lines=15, 
                            interactive=True
                        )
                    
                    with gr.Row():
                         chunk_size_text_input = gr.Number(
                            label="Chunk Size (Characters)", 
                            value=1500, 
                            minimum=100, 
                            maximum=5000,
                            info="Adjust chunk size for long texts (긴 텍스트를 위한 청크 크기 조절)"
                        )
                         text_progress_status = gr.Textbox(label="Progress", value="Ready", interactive=False, lines=1)
                    
                    with gr.Row():
                        paste_text_btn = gr.Button("Paste Input📋", size="sm")
                        translate_btn = gr.Button("Translate Text🚀", variant="primary", size="sm")
                        stop_text_btn = gr.Button("Stop ⏹️", variant="stop", size="sm")
                        download_text_btn = gr.Button("Download Result 💾", size="sm")
                        copy_text_btn = gr.Button("Copy to Clipboard 📋", size="sm")
                    
                    text_download_component = gr.File(label="Download Translated Text (번역 파일 다운로드)", interactive=False, visible=False)
                    


                    # Text Translation Event
                    translate_btn.click(fn=lambda: gr.update(visible=False), inputs=None, outputs=text_download_component)

                    text_translate_evt = translate_btn.click(
                        fn=translate_streaming,
                        inputs=[
                            input_text, 
                            from_lang_drop, 
                            to_lang_drop, 
                            model_name_input, 
                            temp_slider,
                            base_url_input,
                            chunk_size_text_input
                        ],
                        outputs=[output_text, text_progress_status]
                    )
                    
                    stop_text_btn.click(fn=None, inputs=None, outputs=None, cancels=[text_translate_evt])
                    
                    download_text_btn.click(
                        fn=save_translation_to_file,
                        inputs=output_text,
                        outputs=text_download_component
                    )
                    
                    copy_text_btn.click(
                        fn=None,
                        inputs=output_text,
                        outputs=None,
                        js="(text) => { navigator.clipboard.writeText(text); }"
                    )
                    
                    paste_text_btn.click(
                        fn=get_clipboard_text,
                        inputs=None,
                        outputs=input_text
                    )

                # TAB 2: File Translation
                with gr.TabItem("📂 File Translation (파일 번역)"):
                    gr.Markdown("### Upload a text file to translate it chunk by chunk.")
                    gr.Markdown("Supported formats: .txt, .md, .py, .js, .html, etc. (UTF-8 encoded text files)")
                    
                    with gr.Row():
                        file_input = gr.File(
                            label="Upload Text File (텍스트 파일 업로드)", 
                            file_types=[".txt", ".md", ".py", ".js", ".html", ".json", ".csv"]
                        )
                    
                    with gr.Row():
                         chunk_size_input = gr.Number(
                            label="Chunk Size (Characters)", 
                            value=1500, 
                            minimum=100, 
                            maximum=5000,
                            info="Adjust chunk size to fit model context window."
                        )
                    
                    with gr.Row():
                        translate_file_btn = gr.Button("Translate File (파일 번역하기) 📂", variant="primary", size="sm")
                        stop_file_btn = gr.Button("Stop (중단) ⏹️", variant="stop", size="sm")
                    
                    with gr.Row():
                        file_preview = gr.Textbox(
                            label="Translation Preview (번역 미리보기 - 실시간 업데이트)", 
                            lines=15, 
                            interactive=False
                        )
                        download_file = gr.File(label="Download Translated File (번역 파일 다운로드)")
                        copy_file_btn = gr.Button("Copy Preview (미리보기 복사) 📋")

                    # File Translation Event
                    file_translate_evt = translate_file_btn.click(
                        fn=translate_file_process,
                        inputs=[
                            file_input, 
                            from_lang_drop, 
                            to_lang_drop, 
                            model_name_input, 
                            temp_slider, 
                            base_url_input,
                            chunk_size_input
                        ],
                        outputs=[file_preview, download_file]
                    )
                    
                    stop_file_btn.click(fn=None, inputs=None, outputs=None, cancels=[file_translate_evt])
                    
                    copy_file_btn.click(
                        fn=None,
                        inputs=file_preview,
                        outputs=None,
                        js="(text) => { navigator.clipboard.writeText(text); }"
                    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
