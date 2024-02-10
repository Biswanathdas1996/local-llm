from flask import Flask, jsonify, request
# from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne
# from langchain.llms import OpenAI



def getResponse():
    model_path = "./models/mistral-7b-openorca.Q4_0.gguf" 
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
        # temperature=0.6
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question = "World largest forest?"
    print(llm_chain.run(question))
    return llm_chain.run(question)


# Initialize Flask app
app = Flask(__name__)

# Define API endpoints
@app.route('/ask', methods=['GET'])
def get_todos():
    return jsonify(getResponse())



# Start server
if __name__ == '__main__':
    app.run(debug=True)
