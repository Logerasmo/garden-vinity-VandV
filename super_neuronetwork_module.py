from ollama import Client

client = Client(host="http://127.0.0.1:11435")

system_prompt = Path("extraction_prompt.txt").read_text(encoding="utf-8")
prompt = "Кратко объясни, что такое КПД солнечного элемента."

result = client.generate(
    model="qwen3.5:9b",
    prompt=prompt,
    think=False,
    stream=False,
)

print(result.response)