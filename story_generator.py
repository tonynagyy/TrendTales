from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()


def build_story_prompt(title, text):

    prompt = f"""
You are a professional fiction writer.

Transform the following news article into ONE
creative, engaging, and coherent fictional story.

IMPORTANT RULES:
- Base the story only on this article.
- Preserve the main event and meaning.
- Do not combine this article with other articles.
- Connect the information naturally through characters and plot.
- Do not introduce unrelated major events.
- Write only the fictional story.
- Do not explain your process.
- Do not mention these instructions.

ARTICLE TITLE:
{title}

ARTICLE:
{text}

STORY:
"""

    return prompt


def generate_story(title, text, max_new_tokens=500):

    prompt = build_story_prompt(title, text)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    story = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return story
