import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import Dict, Any
import os

class DeepSeekOCR:
    """
    Uses DeepSeek-VL (Vision Language Model) for OCR on scanned documents and images.
    """
    
    def __init__(self, model_path: str = "deepseek-ai/deepseek-vl-7b-chat", device: str = "cuda"):
        self.device = device
        # Note: In a production environment, you might want to call this via an API (vLLM) 
        # instead of loading the model directly to save memory if vLLM is already running it.
        # For this implementation, we'll assume direct loading or a placeholder for API call.
        
        # Check if we are running in a lightweight mode (e.g. CI/CD) where we don't want to load the model
        self.mock_mode = os.getenv("MOCK_OCR", "false").lower() == "true"
        
        if not self.mock_mode:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.vl_chat_processor = self.tokenizer.vl_chat_processor
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16
                ).to(self.device)
            except Exception as e:
                print(f"Warning: Could not load DeepSeek-VL model: {e}. Falling back to mock mode.")
                self.mock_mode = True

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Performs OCR on a single image.
        """
        if self.mock_mode:
            return {
                "text": f"[Mock OCR] Content of {image_path}",
                "metadata": {"origin": "deepseek-ocr-mock"}
            }

        try:
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>Transcribe the text in this image exactly as it appears, preserving layout where possible.",
                    "images": [image_path]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]

            pil_image = Image.open(image_path).convert("RGB")
            
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=[pil_image],
                force_batchify=True
            ).to(self.device)

            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=2048,
                do_sample=False,
                use_cache=True
            )

            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            return {
                "text": answer,
                "metadata": {
                    "filename": os.path.basename(image_path),
                    "origin": "deepseek-ocr"
                }
            }
            
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "metadata": {"origin": "deepseek-ocr", "status": "failed"}
            }
