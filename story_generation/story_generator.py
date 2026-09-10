import os
import re
from typing import List, Dict, Any, Optional
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

# STORY PROMPT BUILDER
def build_story_prompt(events: List[Dict[str, Any]]) -> str:

    events_text = ""

    for i, event in enumerate(events, 1):
        title = event.get("title", "").strip()
        content = event.get("content", "").strip()
        source = event.get("source", "").strip()

        events_text += f"""
EVENT {i}

Title:
{title}

Description:
{content}

Source:
{source}

------------------------
"""

    prompt = f"""You are a professional fiction writer.

Your task is to transform the following real-world trending events into ONE
creative, engaging, and coherent fictional story.

IMPORTANT RULES:
1. Include ALL events provided.
2. Do not omit important events.
3. Preserve the main meaning of every event.
4. Connect all events naturally through characters and plot.
5. Do not introduce unrelated major events.
6. Create consistent characters and locations.
7. Do not change the meaning of the original events.
8. The final output must be a fictional narrative story.
9. Write ONLY the story.
10. Do not mention that you were given events or instructions.

TRENDING EVENTS:
{events_text}

Now write one coherent fictional story that naturally connects all events.
"""
    return prompt


# STORY GENERATION ENGINE
def generate_story_via_hf_api(prompt: str, token: str, model_name: str = "Qwen/Qwen2.5-72B-Instruct") -> str:
    """
    Generates story using Hugging Face's Free Inference API.
    Fast, high-quality, and requires 0 local GPU memory.
    """
    # pyrefly: ignore [missing-import]
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=token)

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9,
    )

    story = response.choices[0].message.content.strip()
    return story


def generate_story_local(prompt: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
    """
    Offline local generation fallback using PyTorch and Transformers.
    """
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[info] Loading local story model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=900,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    story = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return story


def generate_story_from_events(
    events: List[Dict[str, Any]],
    max_new_tokens: int = 1000,
    model_name: Optional[str] = None
) -> str:
    """
    Main Story Generation pipeline function:
    Takes top trending events from RAG and synthesizes them into ONE coherent fictional narrative story.
    """
    if not events:
        raise ValueError("No events provided to generate story.")

    prompt = build_story_prompt(events)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    # Method 1: Hugging Face API (Free, Flagship Qwen2.5-72B, 0 VRAM usage)
    if hf_token:
        try:
            target_model = model_name or "Qwen/Qwen2.5-72B-Instruct"
            print(f"[info] Generating story via Hugging Face API ({target_model})...")
            return generate_story_via_hf_api(prompt, token=hf_token, model_name=target_model)
        except Exception as e:
            print(f"[warn] HF API generation failed ({e}). Falling back to local generation...")

    # Method 2: Local Model
    target_local_model = model_name or "Qwen/Qwen2.5-1.5B-Instruct"
    return generate_story_local(prompt, model_name=target_local_model)


if __name__ == "__main__":
    sample_events = [
        {
            "title": "NASA announces new deep space rover mission",
            "content": "Engineers and scientists have officially unveiled a next-generation rover to explore subsurface ice.",
            "source": "SpaceNews"
        },
        {
            "title": "Global coffee crop threatened by unexpected heatwaves",
            "content": "Record high temperatures across high-altitude farms are pushing farmers to innovate rapidly.",
            "source": "AgriDaily"
        }
    ]
    print("Testing story generation...")
    story_result = generate_story_from_events(sample_events)
    print("\n--- GENERATED STORY ---")
    print(story_result)
