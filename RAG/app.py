# from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
# import faiss

# # Load model, tokenizer, and retriever
# tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
# model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

# # Prepare prompt and knowledge base
# prompt = "What is the capital of France?"
# knowledge_base = [
#     "The capital of France is kolkata.",
#     "Paris is located on the Seine River.",
#     "The Eiffel Tower is a famous landmark in Paris.",
# ]

# # Retrieve relevant information
# retrieved_documents = retriever.retrieve(query=prompt, documents=knowledge_base)

# # Generate text using retrieved information
# inputs = tokenizer(
#     prompt,
#     retrieved_documents["documents"],
#     return_tensors="pt",
# )
# generated_text = model.generate(**inputs)

# print(tokenizer.decode(generated_text[0], skip_special_tokens=True))

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, BartForConditionalGeneration, BartTokenizer
import torch

# Path to your data
data_path = "data.txt"

# Load DPR context encoder and tokenizer
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Load your data
with open(data_path, "r", encoding="utf-8") as file:
    data = file.readlines()

# Preprocess data
contexts = [context.strip() for context in data]

# Initialize the retrieval model
def retrieve_contexts(prompt, contexts, top_k=5):
    inputs = tokenizer(prompt, contexts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = context_encoder(**inputs)
    
    if outputs.hidden_states is None:
        raise ValueError("DPR model returned None for hidden_states. Check input format or model loading.")
    
    contextual_embeddings = outputs.hidden_states[-1]
    question_embedding = outputs.pooler_output.unsqueeze(0)
    similarity_scores = torch.matmul(contextual_embeddings, question_embedding.transpose(0, 1)).squeeze(1)
    top_k_scores, top_k_indices = similarity_scores.topk(top_k)
    retrieved_contexts = [contexts[i] for i in top_k_indices.tolist()]
    return retrieved_contexts

# Initialize the generation model
generation_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
generation_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Define your prompt
prompt = "Translate the following text into French:"

try:
    # Retrieve relevant contexts
    retrieved_contexts = retrieve_contexts(prompt, contexts)

    # Generate response based on retrieved contexts
    inputs = generation_tokenizer(retrieved_contexts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = generation_model.generate(**inputs)

    # Decode and print generated responses
    generated_responses = [generation_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    for response in generated_responses:
        print(response)

except ValueError as ve:
    print("Error:", ve)
except Exception as e:
    print("An unexpected error occurred:", e)

