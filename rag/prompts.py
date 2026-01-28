from pathlib import Path

def load_prompt(name: str) -> str:
    p = Path("prompts") / name
    return p.read_text(encoding="utf-8")

def format_prompt(template: str, context: str, question: str) -> str:
    return template.format(context=context, question=question)

