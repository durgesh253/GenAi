from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
# pip install langchain-google-genai langchain


def create_chatbot():
    # Initialize the Gemini model
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key="AIzaSyBeGmYdFeWkG7XzUKMDrFEIemeJJH-4kMc",
        temperature=0.7
    )
    
    # Create a conversation memory
    memory = ConversationBufferMemory()
    
    # Define the chat template
    template = """You are a helpful and knowledgeable assistant.
    
    Current conversation:
    {history}
    
    Human: {input}
    Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # Create the conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    
    return conversation

def chat_loop(conversation):
    print("Chatbot: Hello! I'm your AI assistant. How can I help you today? (Type 'quit' to exit)")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
            
        try:
            response = conversation.predict(input=user_input)
            print("Chatbot:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Chatbot: I apologize, but I encountered an error. Please try again.")

if __name__ == "__main__":
    # Create and start the chatbot
    chatbot = create_chatbot()
    chat_loop(chatbot)