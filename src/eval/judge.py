"""LLM-as-a-judge: scores a generated answer's faithfulness and relevancy
against its retrieved context, using Groq's free tier (groq/compound-mini).
"""

import json

from groq import Groq

from src.utils.config import GROQ_API_KEY
from src.utils.groq_retry import call_with_retry

JUDGE_MODEL = "groq/compound-mini"

SYSTEM_PROMPT = """You are an impartial judge evaluating a RAG system's answer.

Score two dimensions, each a float from 0.0 to 1.0:
- faithfulness: Is the answer fully grounded in the given context (no hallucinated facts)?
  An honest "I don't know" when the context is insufficient should score 1.0 for faithfulness.
- relevancy: Does the answer actually address the question asked?

Respond ONLY with a JSON object: {"faithfulness": 0.0, "relevancy": 0.0}"""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}

Answer to evaluate: {answer}"""

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def judge_answer(query: str, answer: str, contexts: list[str]) -> dict:
    context_block = "\n\n".join(contexts) if contexts else "(no context retrieved)"
    user = USER_PROMPT_TEMPLATE.format(context=context_block, query=query, answer=answer)
    try:
        resp = call_with_retry(
            lambda: get_client().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=100,
            )
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        return {
            "faithfulness": max(0.0, min(1.0, float(data.get("faithfulness", 0.0)))),
            "relevancy": max(0.0, min(1.0, float(data.get("relevancy", 0.0)))),
        }
    except Exception as e:
        print(f"Judge failed for query '{query[:50]}...': {e}")
        return {"faithfulness": 0.0, "relevancy": 0.0}
