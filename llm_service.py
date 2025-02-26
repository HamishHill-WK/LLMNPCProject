
# llm_service.py - Connection to locally-hosted LLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
from threading import Thread

class LocalLLMService:
    def __init__(self, model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-Q2_K.gguf", device="cpu"):
        """
        Initialize the local LLM service.
        
        Args:
            model_name (str): The model identifier from Hugging Face or local path
            device (str): "cpu" or "cuda" (if GPU is available)
        """
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_name = model_name
        
        # Load model based on available hardware
        self.setup_model()
    
    def setup_model(self):
        """Load the model based on available hardware resources"""
        print(f"Initializing model {self.model_name} on {self.device}...")
        
        try:
            # For GGUF quantized models with llama.cpp (lowest resource usage)
            from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
            
            self.model = CTAutoModelForCausalLM.from_pretrained(
                self.model_name,
                model_type="llama",
                gpu_layers=0 if self.device == "cpu" else 50  # Adjust based on GPU VRAM
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.use_ctransformers = True
            print("Using llama.cpp backend (CTTransformers)")
            
        except (ImportError, ValueError, Exception) as e:
            print(f"Could not load with CTTransformers: {e}")
            print("Falling back to Transformers library")
            
            # Try standard Hugging Face Transformers approach
            try:
                # Use 4-bit or 8-bit quantization to reduce memory usage
                from transformers import BitsAndBytesConfig
                
                if torch.cuda.is_available() and self.device == "cuda":
                    # 4-bit quantization for GPU
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                else:
                    quantization_config = None
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                self.use_ctransformers = False
                
            except Exception as e:
                print(f"Error loading model with Transformers: {e}")
                raise RuntimeError("Could not initialize any model backend")
        
        print(f"Model loaded successfully with {'llama.cpp' if hasattr(self, 'use_ctransformers') and self.use_ctransformers else 'transformers'}")
    
    def generate_response(self, prompt, max_new_tokens=150, temperature=0.7):
        """Generate a response from the local model"""
        try:
            if hasattr(self, 'use_ctransformers') and self.use_ctransformers:
                # CTTransformers generation
                input_text = self._format_prompt_for_model(prompt)
                
                # Generate text with llama.cpp backend
                response = self.model(
                    input_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                # Extract just the assistant's response
                return self._extract_response(input_text, response)
            
            else:
                # Standard Transformers generation
                inputs = self.tokenizer(self._format_prompt_for_model(prompt), return_tensors="pt").to(self.device)
                
                # Create a streamer for non-blocking generation
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                # Start generation in a separate thread
                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "streamer": streamer,
                }
                
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Collect the generated text
                generated_text = ""
                for text in streamer:
                    generated_text += text
                
                return generated_text.strip()
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I cannot respond right now."
    
    def _format_prompt_for_model(self, prompt):
        """Format the prompt based on the model type"""
        # Format for Llama-2-Chat models
        if "llama-2" in self.model_name.lower() and "chat" in self.model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        
        # Format for general instruction models
        return f"### Instruction:\n{prompt}\n\n### Response:"
    
    def _extract_response(self, prompt, full_response):
        """Extract just the model's response part from the full response"""
        # Remove the original prompt to get just the generated text
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            response = full_response
            
        # Remove any additional prompt markers that might have been generated
        end_markers = ["[INST]", "### Instruction:", "<|user|>", "<s>"]
        for marker in end_markers:
            if marker in response:
                response = response.split(marker)[0].strip()
                
        return response

    def get_model_info(self):
        """Return information about the currently loaded model"""
        device_str = f"GPU ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else "CPU"
        backend = "llama.cpp (CTTransformers)" if hasattr(self, 'use_ctransformers') and self.use_ctransformers else "Hugging Face Transformers"
        
        return {
            "model_name": self.model_name,
            "device": device_str,
            "backend": backend,
        }
