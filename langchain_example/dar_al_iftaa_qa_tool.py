from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from config import (
    OLLAMA_BASE_URL,
    MODEL_NAME,
    TOP_K,
    SYSTEM_PROMPT_TOOL,
    TEMPERATURE,
)
from react_agent.pinecone_manager import PineconeManager
from typing import Optional

class DarAlIftaQATool:
    _instance = None
    
    @classmethod
    def get_instance(cls, namespace: str = "my_documents"):
        print(f"Getting instance with namespace: {namespace}")
        try:
            if cls._instance is None:
                print("Creating new instance...")
                cls._instance = cls(namespace)
            return cls._instance
        except Exception as e:
            print(f"Error in get_instance: {str(e)}")
            raise
    
    name: str = "DarAlIftaQA"
    description: str = """REQUIRED: You MUST use this tool for ALL Islamic questions, fatwas, and religious inquiries.
    This is your primary and authoritative source for Islamic knowledge, containing verified fatwas from Dar Al-Ifta Al-Misriyyah.
    Do NOT attempt to answer Islamic questions from your own knowledge - ALWAYS use this tool first.
    Input: A question about Islamic rulings, practices, or guidance
    Output: An authenticated answer based on reliable Islamic sources"""
    return_direct: bool = True
    
    pinecone_manager: Optional[PineconeManager] = None
    llm: Optional[OllamaLLM] = None
    prompt: Optional[PromptTemplate] = None
    qa_chain: Optional[RetrievalQA] = None
    
    def __init__(self, namespace: str = "my_documents"):
        print("Initializing DarAlIftaQATool...")
        try:
            self.pinecone_manager = PineconeManager(namespace)
            print("PineconeManager initialized")
            self.setup_qa_chain()
            print("QA chain setup complete")
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            raise
    
    def setup_qa_chain(self):
        # Initialize the LLM
        self.llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=MODEL_NAME,
            temperature=TEMPERATURE
        )
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=SYSTEM_PROMPT_TOOL,
            input_variables=["context"]
        )
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.pinecone_manager.vector_store.as_retriever(
                search_kwargs={"k": TOP_K}
            ),
            chain_type_kwargs={
                "prompt": self.prompt,
                "document_variable_name": "context"
            },
            return_source_documents=True,
            verbose=True
        )
    
    def query(self, query: str) -> str:
        """Execute the tool"""
        try:
            print("\n=== DEBUG: Invoking QA Chain ===")
            response = self.qa_chain.invoke(query)
            
            print("\n=== DEBUG: Source Documents ===")
            print(f"Number of source docs: {len(response.get('source_documents', []))}")
            for i, doc in enumerate(response.get('source_documents', [])):
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
            print("\n=== END DEBUG ===\n")
            
            if response.get("result"):
                # Get unique sources using a set to remove duplicates
                sources = set(
                    doc.metadata.get('source', 'No source available')
                    for doc in response.get("source_documents", [])
                )
                
                result = f"\nAnswer: {response['result']}"
                if sources:
                    result += "\n\nSources:"
                    for source in sources:
                        result += f"\n- {source}"
                return result
            else:
                return "\nError: No valid response received from the model"
            
        except Exception as e:
            print(f"\n=== DEBUG: Error ===")
            print(f"Exception type: {type(e)}")
            print(f"Exception message: {str(e)}")
            print("\n=== END DEBUG ===\n")
            return f"\nError in DarAlIftaQA tool: {str(e)}"

def qa_tool_func(question: str) -> str:
    qa_instance = DarAlIftaQATool.get_instance()
    response = qa_instance.query(question)
    return response
