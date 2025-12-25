from langchain_groq import ChatGroq
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

class AgentState(TypedDict):
    question: str
    query: str
    docs: List[str]
    is_sufficient: bool
    iterations: int 
    max_iterations: int 
    final_answer: str

class Agentic_RAG:
    def __init__(self, path:str):
        self.model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.loader = PyPDFLoader(path)
        self.splitter = RecursiveCharacterTextSplitter( chunk_size=500, chunk_overlap=80)
        self.parser = JsonOutputParser()
        self.docs_retriever = None
        self.workflow = None

    def ingest_documents(self):
        document = self.loader.load()
        chunks = self.splitter.split_documents(document)
        vector_store = FAISS.from_documents(chunks, embedding=self.embedding)
        vector_store.save_local('vector_store')
        print(f"Documents loaded successfully!")

    def load_vectorstore(self):
        vector_store = FAISS.load_local(
            'vector_store',
            embeddings=self.embedding,
            allow_dangerous_deserialization=True
        )
        self.docs_retriever = vector_store.as_retriever(search_kwargs={'k':4})

    def retrieve(self, state: AgentState):
        docs = self.docs_retriever.invoke(state['question'])
        return {
            'docs': [d.page_content for d in docs],
            'iterations': state['iterations'] + 1
        }
    
    def check_sufficient(self, state: AgentState):
        prompt = PromptTemplate(
            template="""
                Question:
                {question}

                context:
                {context}

                Is the context sufficient to fully answer the question?

                Respond strictly in JSON:
                {{
                "sufficient": true/false,
                "query": "refined search query if insufficient"
                }}
            """,
            input_variables=['question','context']
        )

        chain = prompt | self.model | self.parser
        response = chain.invoke({'question': state['question'], 'context': "\n".join(state["docs"])})

        return {
            'is_sufficient': response['sufficient'],
            'query': response.get('query',"")
        } 
    
    def answer(self, state: AgentState):
        prompt = PromptTemplate(
            template="""
            You are an expert assistant.

            Answer the question in a clear, detailed, elaborate, and well-structured manner
            using ONLY the information provided in the context below.

            Context: {context}
            question: {question}
            """,
            input_variables=['question','context']
        )

        chain = prompt | self.model
        response = chain.invoke({'question': state['question'], 'context': "\n".join(state["docs"])})
        return {'final_answer': response.content}
    
    def should_continue(self, state: AgentState):
        if state['is_sufficient']:
            return 'answer'
        if state['iterations'] >= state['max_iterations']:
            return 'answer'
        return 'retrieve'
    
    def build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node('retrieve', self.retrieve)
        graph.add_node('check_sufficient', self.check_sufficient)
        graph.add_node('answer', self.answer)

        graph.add_edge(START, 'retrieve')
        graph.add_edge('retrieve', 'check_sufficient')
        graph.add_conditional_edges(
            'check_sufficient',
            self.should_continue,
            {
                "retrieve": "retrieve",
                "answer": "answer"
            }
        )
        graph.add_edge('answer', END)

        self.workflow = graph.compile()

    def output_result(self, question:str):
        if not self.workflow:
            self.build_graph()

        initial_state: AgentState = {
            "question": question,
            "query": "",
            "docs": [],
            "is_sufficient": False,
            "iterations": 0,
            "max_iterations": 5,
            "final_answer": ""
        }

        result = self.workflow.invoke(initial_state)
        return result["final_answer"]
    