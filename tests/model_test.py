from llama_cpp import Llama

llm = Llama(model_path="model/mistral-7b-instruct-v0.2.Q3_K_M.gguf")

output = llm("Explain what Retrieval-Augmented Generation (RAG) is in one sentence.")
print(output["choices"][0]["text"])
