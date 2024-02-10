# from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne
# from langchain.llms import OpenAI

# Make sure the model path is correct for your system!
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
    verbose=False,
    # temperature=0.6
)

# Uncomment the code below if you want to run inference on CPU
# llm = LlamaCpp(
#     model_path=model_path, callback_manager=callback_manager, verbose=True
# )

llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = """
# Context:=>2600

#     Title: Unveiling the Magnificence: A Journey Through the Pyramids of Egypt

#     Introduction:
#     The Pyramids of Egypt stand as enduring monuments to the ingenuity, power, and mystique of one of the world's most ancient civilizations. These colossal structures, shrouded in centuries of myth and legend, continue to captivate the imagination of people across the globe. As we embark on a journey to explore the Pyramids of Egypt, we delve into their history, significance, construction, and enduring legacy.

#     Historical Context:
#     Constructed over 4,500 years ago during the Old Kingdom period of ancient Egypt, the pyramids were built as royal tombs for pharaohs, serving as monumental structures to house their mummified bodies and treasures for the afterlife. The most famous among them are the Great Pyramid of Giza, the Pyramid of Khafre, and the Pyramid of Menkaure, located on the Giza Plateau near Cairo.

#     Architectural Marvels:
#     The Pyramids of Egypt are architectural marvels, constructed with precision and skill that continue to baffle modern engineers and architects. Composed primarily of limestone and granite blocks, these structures were meticulously built using advanced engineering techniques for their time. The Great Pyramid of Giza, attributed to Pharaoh Khufu, stands as the largest and most iconic of all, reaching a height of over 480 feet and consisting of approximately 2.3 million stone blocks.

# from the above  context find the correct answer of 

# when the pirimed was Constructed ?

# # Instruction
# ## do not provide any info outside of the context.
# """
# question = """
# संदर्भ डेटा: मिस्र के पिरामिडों के माध्यम से एक यात्रा
# पिरामिडों का निर्माण 2500-2300 ईसा पूर्व के बीच हुआ था। मिस्र के पिरामिड दुनिया की सबसे प्राचीन सभ्यताओं में से एक की प्रतिभा, शक्ति और रहस्य के स्थायी स्मारक के रूप में खड़े हैं। सदियों के मिथक और किंवदंतियों में डूबी ये विशाल संरचनाएं, दुनिया भर के लोगों की कल्पना को आकर्षित करती रहती हैं।

# पिरामिड किसने बनाया?

# #निर्देश
# ## केवल संदर्भ के आधार पर उत्तर दें, यदि नहीं तो कहें कि मुझे नहीं पता
# """

question = "World largest forest?"


print(llm_chain.run(question))

