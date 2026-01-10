from langchain_groq import ChatGroq
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
import os
from langgraph.graph.message import BaseMessage, add_messages
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    docs: List[str]
    needs_retrieval: bool
    iterations: int 
    max_iterations: int 
    final_answer: str

class Agentic_RAG:
    def __init__(self, path: str):
        api_key = os.environ.get('PINECONE_API_KEY')
        self.model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.loader = PyPDFLoader(path)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=80)
        self.parser = JsonOutputParser()
        self.workflow = None
        self.pc = Pinecone(api_key=api_key)
        self.index_name = 'self-corrective-rag'
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.pc.Index(self.index_name)

    def ingest_documents(self):
        try:
            document = self.loader.load()
            chunks = self.splitter.split_documents(document)
            texts = [doc.page_content for doc in chunks]
            embeddings = self.embedding.embed_documents(texts=texts)   
            vectors = []

            for i, text in enumerate(texts):
                vectors.append({
                    'id': f'chunk-{i}',
                    'values': embeddings[i],
                    'metadata': {
                        'chunk_id': i,
                        'text': text
                    }
                })

            self.index.upsert(vectors=vectors)
        except:
            return "There is an error while uploading the document! Please try again!"

    def decide_retrieval(self, state: AgentState):
        try:
            prompt = PromptTemplate(
                template="""
                You are a smart assistant that decides whether to retrieve documents or answer from your knowledge.

                Question:
                {question}

                Conversation so far:
                {history}

                Based on the question, decide if you need to retrieve documents from a knowledge base to answer accurately.
                - Answer "yes" if the question requires specific information from documents
                - Answer "no" if you can answer from general knowledge

                Respond ONLY in JSON format:
                {{"needs_retrieval": true/false, "reasoning": "brief explanation"}}
                """,
                input_variables=['question', 'history']
            )

            history = "\n".join([f'{m.type}: {m.content}' for m in state['messages'][:-1]]) if len(state['messages']) > 1 else "None"
            
            chain = prompt | self.model | self.parser
            response = chain.invoke({
                'question': state['messages'][-1].content,
                'history': history
            })

            return {
                'needs_retrieval': response['needs_retrieval'],
                'iterations': state['iterations'] + 1
            }
        except:
            return {
                'needs_retrieval': True,
                'iterations': state['iterations'] + 1
            }
           
    def retrieve(self, state: AgentState):
        try:
            last_message = state['messages'][-1].content
            query = state['query'] or last_message
            embed_query = self.embedding.embed_query(query)

            result = self.index.query(
                vector=embed_query,
                top_k=5,
                include_metadata=True
            )
            docs = [m['metadata']['text'] for m in result['matches']]
            
            return {
                'docs': docs,
                'iterations': state['iterations'] + 1,
                'messages': [
                    SystemMessage(content=f"Retrieved content: \n{docs}")
                ]
            }
        except:
            return "There is an information retrieve error. Please try again!"
    
    def check_sufficient(self, state: AgentState):
        try:
            prompt = PromptTemplate(
                template="""
                Conversation so far:
                {history}

                Question:
                {question}

                Retrieved Context:
                {context}

                Is the retrieved context sufficient to fully answer the question?

                Respond strictly in JSON:
                {{"sufficient": true/false, "refined_query": "refined search query if insufficient"}}
                """,
                input_variables=['history', 'question', 'context']
            )

            history = "\n".join([f'{m.type}: {m.content}' for m in state['messages'][:-1]]) if len(state['messages']) > 1 else "None"

            chain = prompt | self.model | self.parser
            response = chain.invoke({
                'history': history,
                'question': state['messages'][-1].content, 
                'context': "\n".join(state["docs"])
            })

            return {
                'is_sufficient': response['sufficient'],
                'query': response.get('refined_query', "")
            } 
        except:
            return {
                'is_sufficient': True,
                'query': ''
            } 

    
    def answer(self, state: AgentState):
        try:
            if state['docs']:
                prompt = PromptTemplate(
                    template="""
                    You are an expert assistant.

                    Answer the question in a clear, detailed, elaborate, and well-structured manner
                    using ONLY the information provided in the context below.

                    Conversation so far:
                    {history}

                    Context: 
                    {context}

                    Question: 
                    {question}
                    """,
                    input_variables=['history', 'question', 'context']
                )
                context = "\n".join(state["docs"])
            else:
                prompt = PromptTemplate(
                    template="""
                    You are an expert assistant.

                    Answer the following question clearly and accurately using your general knowledge.

                    Conversation so far:
                    {history}

                    Question: 
                    {question}
                    """,
                    input_variables=['history', 'question']
                )
                context = ""

            history = "\n".join([f'{m.type}: {m.content}' for m in state['messages'][:-1]]) if len(state['messages']) > 1 else "None"
            
            if state['docs']:
                chain = prompt | self.model
                response = chain.invoke({
                    'history': history,
                    'question': state['messages'][-1].content, 
                    'context': context
                })
            else:
                chain = prompt | self.model
                response = chain.invoke({
                    'history': history,
                    'question': state['messages'][-1].content
                })

            return {
                'final_answer': response.content,
                'messages': [response]
            }
        except:
            return {
                'final_answer': '',
                'messages': ['']
            }
    
    def should_continue(self, state: AgentState):
        """Decide next step: retrieve -> check -> answer or answer"""
        if state['iterations'] >= state['max_iterations']:
            return 'answer'
        
        if state.get('is_sufficient', False):
            return 'answer'
        
        if state.get('needs_retrieval', False):
            return 'answer'
        
        return 'check_sufficient'
    
    def build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node('decide_retrieval', self.decide_retrieval)
        graph.add_node('retrieve', self.retrieve)
        graph.add_node('check_sufficient', self.check_sufficient)
        graph.add_node('answer', self.answer)

        graph.add_edge(START, 'decide_retrieval')

        graph.add_conditional_edges(
            'decide_retrieval',
            lambda state: 'retrieve' if state['needs_retrieval'] else 'answer',
            {
                'retrieve': 'retrieve',
                'answer': 'answer'
            }
        )
        
        graph.add_edge('retrieve', 'check_sufficient')
        
        graph.add_conditional_edges(
            'check_sufficient',
            self.should_continue,
            {
                'retrieve': 'retrieve',
                'check_sufficient': 'check_sufficient',
                'answer': 'answer'
            }
        )
        
        graph.add_edge('answer', END)

        self.workflow = graph.compile()

    def output_result(self, question: str):
        if not self.workflow:
            self.build_graph()

        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "query": "",
            "docs": [],
            "needs_retrieval": False,
            "iterations": 0,
            "max_iterations": 5,
            "final_answer": ""
        }

        result = self.workflow.invoke(initial_state)
        return result["final_answer"]
