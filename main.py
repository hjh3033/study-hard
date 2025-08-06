import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel # FastAPI에서 요청 본문을 처리하기 위해 추가
from typing import List, Tuple # 타입 힌트를 위해 추가

# RAG 관련 라이브러리 (이전과 동일)
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # MessagesPlaceholder 추가
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
# --- 대화 기록(Memory)을 위한 라이브러리 ---
from langchain_core.messages import AIMessage, HumanMessage

# --- 1. RAG 파이프라인 설정 (이전과 동일) ---
loader = TextLoader("knowledge.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(docs)
embeddings = OllamaEmbeddings(model="llama3.1:8b-instruct-q4_0")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()


# --- 2. 도구 정의 (이전과 동일) ---
@tool
def patent_searcher(query: str) -> str:
    """'특허'와 관련된 외부 정보를 검색할 때 사용하는 도구입니다."""
    print(f"Tool Executed: patent_searcher")
    return "외부 특허 DB 검색 결과: '자동 장력 조절 시스템' 관련 특허가 다수 존재합니다."

@tool
def internal_knowledge_searcher(query: str) -> str:
    """
    'VB-Standard', 'VB-HeatResist', 'VB-PowerPlus' 등 특정 모델명이나,
    '고온', '내열', '고장력' 등 제품의 기술 사양, 특징, 재질과 관련된 내부 정보를 찾을 때 사용하는 도구입니다.
    """
    print(f"Tool Executed: internal_knowledge_searcher")
    retrieved_docs = retriever.invoke(query)
    doc_contents = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return f"내부 기술 문서에서 다음 정보를 찾았습니다:\n---\n{doc_contents}"


# --- 3. LLM 엔진과 '기억력'을 갖춘 에이전트 설정 ---
llm = ChatOllama(model="llama3.1:8b-instruct-q4_0") 
tools = [patent_searcher, internal_knowledge_searcher]

# 3-1. (핵심 변경) 프롬프트에 'chat_history'를 기억할 공간을 추가합니다.
# MessagesPlaceholder는 이 자리에 대화 기록이 들어올 것이라고 알려주는 역할을 합니다.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You must use the provided tools to answer the user's questions. Prioritize using the 'internal_knowledge_searcher' for specific product or technical questions."),
    MessagesPlaceholder(variable_name="chat_history"), # 대화 기록이 들어올 자리
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
# 참고: AgentExecutor는 chat_history 입력을 자동으로 처리합니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- 4. FastAPI 서버 설정 ---
app = FastAPI()

# 4-1. (핵심 변경) 프론트엔드에서 보낼 요청 데이터의 형식을 미리 정의합니다.
# 이제는 단순한 query가 아니라, 사용자의 현재 질문(input)과 이전 대화 기록(chat_history)을 함께 받습니다.
class ChatRequest(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] # 예: [("user", "안녕"), ("ai", "안녕하세요!")]

@app.post("/ask") # GET 방식에서 POST 방식으로 변경하여 더 많은 데이터를 받을 수 있게 합니다.
async def ask_endpoint(request: ChatRequest):
    """사용자의 질문과 대화 기록을 받아 LangChain 에이전트를 실행하고 결과를 반환합니다."""
    
    # 4-2. 프론트에서 받은 대화 기록을 LangChain이 이해하는 형식으로 변환합니다.
    chat_history_messages = []
    for user_msg, ai_msg in request.chat_history:
        chat_history_messages.append(HumanMessage(content=user_msg))
        chat_history_messages.append(AIMessage(content=ai_msg))

    print(f"Received input: {request.input}")
    print(f"Chat History: {chat_history_messages}")

    # 4-3. 에이전트를 실행할 때, 현재 질문(input)과 변환된 대화 기록(chat_history)을 함께 전달합니다.
    response = await agent_executor.ainvoke({
        "input": request.input,
        "chat_history": chat_history_messages
    })
    return {"response": response['output']}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
