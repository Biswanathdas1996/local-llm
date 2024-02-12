from flask import Flask, jsonify, request
# from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne
# from langchain.llms import OpenAI



def getResponse(question):
    model_path = "./models/mistral-7b-openorca.Q2_K.gguf" 
    template = """Question: {question}
    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,
        temperature=0.1
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

  
    print(llm_chain.run(question))
    return llm_chain.run(question)


# Initialize Flask app
app = Flask(__name__)

# Define API endpoints
@app.route('/ask', methods=['POST'])
def get_todos():
    question = request.json.get("prompt")
    return jsonify({
        "prompt":question,
        "result":getResponse(question)
    })



# Start server
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
