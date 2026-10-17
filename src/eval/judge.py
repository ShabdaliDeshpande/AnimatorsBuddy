"""Local Ollama model as a DeepEval judge (no OpenAI / cloud calls).

DeepEval metrics call generate(prompt, schema=PydanticModel); we pass the
schema to Ollama as a JSON-schema constraint and return a filled instance.
"""

import asyncio

from deepeval.models import DeepEvalBaseLLM

from src.eval.ollama_json import MODEL, generate_json


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model=MODEL):
        self.model_name = model
        super().__init__(model)

    def load_model(self):
        return self

    def get_model_name(self):
        return f"ollama/{self.model_name}"

    def supports_structured_outputs(self):
        return True

    def supports_json_mode(self):
        return True

    def generate(self, prompt, schema=None, **kwargs):
        if schema is None:
            result = generate_json(prompt, {"type": "object", "properties": {
                "text": {"type": "string"}}, "required": ["text"]}, model=self.model_name)
            return result["text"]
        data = generate_json(prompt, schema.model_json_schema(), model=self.model_name)
        return schema.model_validate(data)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return await asyncio.to_thread(self.generate, prompt, schema=schema)
