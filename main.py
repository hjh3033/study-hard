import os
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
import logging

# LangChain 관련 임포트 (오류 처리 포함)
try:
    from langchain_ollama import ChatOllama
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain.tools import Tool
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain 모듈 임포트 오류: {e}")
    LANGCHAIN_AVAILABLE = False

# 벡터스토어 관련 임포트 (선택적)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    VECTORSTORE_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        VECTORSTORE_AVAILABLE = True
    except ImportError:
        print("Warning: 벡터스토어 모듈을 사용할 수 없습니다. 내부 문서 검색 기능이 제한됩니다.")
        VECTORSTORE_AVAILABLE = False

# KoNLPy 형태소 분석기 추가
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    print("Warning: KoNLPy not installed. 규칙 기반 키워드 추출을 사용합니다.")
    KONLPY_AVAILABLE = False

# KIPRIS API 모듈 임포트
from kipris_api import search_and_extract_patents

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class QueryRequest(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = []

class KoreanKeywordExtractor:
    """한국어 키워드 추출을 위한 클래스"""
    
    def __init__(self):
        self.okt = None
        self.konlpy_available = KONLPY_AVAILABLE  # 글로벌 변수를 인스턴스 변수로 저장
        
        if self.konlpy_available:
            try:
                # Okt가 가장 안정적이고 빠름
                self.okt = Okt()
                logger.info("KoNLPy Okt 형태소 분석기 초기화 완료")
            except Exception as e:
                logger.warning(f"KoNLPy 초기화 실패: {e}")
                self.konlpy_available = False
        
        # 기술 분야별 핵심 키워드 사전
        self.tech_keywords = {
            'belt': ['벨트', '타이밍벨트', 'V벨트', '컨베이어벨트', '전동벨트', '체인', '동력전달'],
            'conveyor': ['컨베이어', '이송', '운반', '반송', '컨베이어시스템'],
            'automation': ['자동화', '로봇', '그리퍼', '픽앤플레이스', '매니퓰레이터'],
            'tension': ['장력', '텐션', '조절', '제어', '모니터링'],
            'material': ['고무', '실리콘', '카본', '케블라', '복합소재', '내열'],
            'system': ['시스템', '장치', '기구', '메커니즘', '구동']
        }
        
        # 불용어 리스트
        self.stopwords = {
            '것', '수', '때', '이', '그', '저', '의', '를', '을', '가', '이', '에', '에서', 
            '로', '으로', '와', '과', '도', '만', '까지', '부터', '한테', '에게', '께',
            '하다', '되다', '있다', '없다', '이다', '아니다', '같다', '다르다',
            '좀', '더', '가장', '매우', '정말', '진짜', '완전', '아주', '너무',
            '찾다', '검색', '알려주다', '보여주다', '주다', '해주다', '드리다'
        }
    
    def extract_keywords_with_morphology(self, text: str) -> List[str]:
        """형태소 분석을 통한 키워드 추출"""
        if not self.okt:
            return self.extract_keywords_with_rules(text)
        
        try:
            # 명사와 복합명사 추출
            nouns = self.okt.nouns(text)
            
            # 2글자 이상의 명사만 필터링
            filtered_nouns = [noun for noun in nouns 
                            if len(noun) >= 2 and noun not in self.stopwords]
            
            # 기술 키워드 우선순위 부여
            priority_keywords = []
            general_keywords = []
            
            for noun in filtered_nouns:
                is_tech_keyword = False
                for category, keywords in self.tech_keywords.items():
                    if any(keyword in noun or noun in keyword for keyword in keywords):
                        priority_keywords.append(noun)
                        is_tech_keyword = True
                        break
                
                if not is_tech_keyword:
                    general_keywords.append(noun)
            
            # 우선순위 키워드를 앞에 배치
            result = priority_keywords + general_keywords
            
            # 중복 제거하면서 순서 유지
            seen = set()
            unique_keywords = []
            for keyword in result:
                if keyword not in seen:
                    seen.add(keyword)
                    unique_keywords.append(keyword)
            
            return unique_keywords[:5]  # 최대 5개까지
            
        except Exception as e:
            logger.warning(f"형태소 분석 실패, 규칙 기반으로 fallback: {e}")
            return self.extract_keywords_with_rules(text)
    
    def extract_keywords_with_rules(self, text: str) -> List[str]:
        """규칙 기반 키워드 추출 (형태소 분석기 없을 때 fallback)"""
        
        # 1. 정규식을 통한 기본 키워드 패턴 추출
        patterns = [
            r'([가-힣]{2,}벨트)',  # ~벨트
            r'([가-힣]{2,}컨베이어)', # ~컨베이어  
            r'([가-힣]{2,}로봇)',    # ~로봇
            r'([가-힣]{2,}시스템)',  # ~시스템
            r'([가-힣]{2,}장치)',    # ~장치
            r'([가-힣]{2,}기구)',    # ~기구
        ]
        
        keywords = []
        
        # 패턴 매칭
        for pattern in patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)
        
        # 2. 기술 키워드 사전과 매칭
        for category, tech_words in self.tech_keywords.items():
            for tech_word in tech_words:
                if tech_word in text:
                    keywords.append(tech_word)
        
        # 3. 한글 명사형 단어 추출 (2-6글자)
        korean_words = re.findall(r'[가-힣]{2,6}', text)
        filtered_words = [word for word in korean_words 
                         if word not in self.stopwords and len(word) >= 2]
        
        keywords.extend(filtered_words)
        
        # 중복 제거 및 우선순위 정렬
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:5]

# 전역 키워드 추출기 인스턴스
keyword_extractor = KoreanKeywordExtractor()

def extract_search_keywords(user_input: str) -> str:
    """사용자 입력에서 검색용 키워드를 추출하는 도구"""
    
    logger.info(f"키워드 추출 시작: {user_input}")
    
    # 1. 형태소 분석 또는 규칙 기반 키워드 추출
    if keyword_extractor.konlpy_available:
        keywords = keyword_extractor.extract_keywords_with_morphology(user_input)
        method = "형태소분석"
    else:
        keywords = keyword_extractor.extract_keywords_with_rules(user_input)
        method = "규칙기반"
    
    # 2. 숫자 정보 추출 (몇 건, 몇 개 등)
    number_match = re.search(r'(\d+)\s*(?:건|개|개|가지|종류)', user_input)
    num_results = int(number_match.group(1)) if number_match else 5
    
    # 키워드를 공백으로 연결
    search_query = ' '.join(keywords) if keywords else user_input
    
    logger.info(f"추출된 키워드 ({method}): {keywords}")
    logger.info(f"생성된 검색쿼리: {search_query}")
    
    result = {
        "search_query": search_query,
        "keywords": keywords,
        "num_results": num_results,
        "extraction_method": method
    }
    
    return json.dumps(result, ensure_ascii=False)

def search_kipris_patents(query_info: str) -> str:
    """개선된 KIPRIS 특허 검색 도구 - REST_API.py 기반 키 매핑"""
    
    try:
        # query_info는 JSON 문자열이므로 파싱
        query_data = json.loads(query_info)
        search_query = query_data.get("search_query", "")
        num_results = query_data.get("num_results", 5)
        keywords = query_data.get("keywords", [])
        method = query_data.get("extraction_method", "")
        
        if not search_query.strip():
            return "검색할 키워드가 없습니다. 다른 검색어를 시도해주세요."
        
        logger.info(f"KIPRIS 검색 실행: '{search_query}' ({num_results}건)")
        
        # KIPRIS API 호출
        patents = search_and_extract_patents(search_query, num_results)
        
        if not patents:
            # 키워드가 있다면 개별적으로 재시도
            if keywords:
                for keyword in keywords[:3]:  # 상위 3개 키워드로 재시도
                    logger.info(f"대체 검색 시도: {keyword}")
                    patents = search_and_extract_patents(keyword, num_results)
                    if patents:
                        search_query = keyword  # 성공한 키워드로 업데이트
                        break
        
        if not patents:
            return f"'{search_query}' 관련 특허를 찾을 수 없습니다. 다른 검색어를 시도해보세요."
        
        # 결과 포맷팅 (REST_API.py 스타일)
        result = f"🔍 **검색어**: {search_query} ({method})\n"
        result += f"📄 **검색 결과**: {len(patents)}건\n\n"
        
        for i, patent in enumerate(patents, 1):
            if isinstance(patent, dict) and 'error' not in patent:
                # REST_API.py와 동일한 키 사용
                title = patent.get('발명명칭', 'N/A')
                applicant = patent.get('출원인', 'N/A')
                app_number = patent.get('출원번호', 'N/A')
                app_date = patent.get('출원일자', 'N/A')
                pub_number = patent.get('공개번호', 'N/A')
                pub_date = patent.get('공개일자', 'N/A')
                reg_number = patent.get('등록번호', 'N/A')
                reg_date = patent.get('등록일자', 'N/A')
                reg_status = patent.get('등록상태', 'N/A')
                abstract = patent.get('초록', 'N/A')
                
                result += f"**[{i}] {title}**\n"
                result += f"   • 출원인: {applicant}\n"
                result += f"   • 출원번호: {app_number} (출원일: {app_date})\n"
                
                if pub_number != 'N/A':
                    result += f"   • 공개번호: {pub_number} (공개일: {pub_date})\n"
                
                if reg_number != 'N/A':
                    result += f"   • 등록번호: {reg_number} (등록일: {reg_date})\n"
                
                result += f"   • 등록상태: {reg_status}\n"
                
                if abstract != 'N/A' and len(abstract) > 150:
                    abstract = abstract[:150] + "..."
                result += f"   • 초록: {abstract}\n\n"
            else:
                # 오류가 있는 경우
                logger.error(f"특허 데이터 오류: {patent}")
                result += f"**[{i}] 데이터 처리 오류**\n\n"
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        return "키워드 추출 결과를 파싱하는 중 오류가 발생했습니다."
    except Exception as e:
        logger.error(f"KIPRIS 검색 중 오류: {e}")
        return f"특허 검색 중 오류가 발생했습니다: {str(e)}"

# RAG 시스템 초기화
def initialize_rag_system():
    """내부 문서 기반 RAG 시스템 초기화"""
    
    if not VECTORSTORE_AVAILABLE:
        logger.warning("벡터스토어를 사용할 수 없어 RAG 시스템을 초기화할 수 없습니다.")
        return None
    
    try:
        # knowledge.txt 파일 읽기
        if os.path.exists("knowledge.txt"):
            with open("knowledge.txt", "r", encoding="utf-8") as f:
                content = f.read()
            
            # 문서 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            docs = [Document(page_content=content)]
            split_docs = text_splitter.split_documents(docs)
            
            # 임베딩 및 벡터스토어 생성
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            
            logger.info("RAG 시스템 초기화 완료")
            return vectorstore
        else:
            logger.warning("knowledge.txt 파일을 찾을 수 없습니다.")
            return None
            
    except Exception as e:
        logger.error(f"RAG 시스템 초기화 실패: {e}")
        return None

# RAG 벡터스토어 초기화
vectorstore = initialize_rag_system()

def search_internal_knowledge(query: str) -> str:
    """내부 지식 문서 검색"""
    
    if not vectorstore:
        # RAG 시스템이 없을 때는 간단한 키워드 매칭으로 대체
        try:
            if os.path.exists("knowledge.txt"):
                with open("knowledge.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 간단한 키워드 매칭
                query_lower = query.lower()
                lines = content.split('\n')
                relevant_lines = []
                
                for line in lines:
                    if any(keyword in line.lower() for keyword in query_lower.split()):
                        relevant_lines.append(line.strip())
                
                if relevant_lines:
                    result = "📚 **내부 지식 문서 검색 결과 (키워드 매칭):**\n\n"
                    for i, line in enumerate(relevant_lines[:5], 1):  # 최대 5개
                        if line:
                            result += f"**[{i}]** {line}\n\n"
                    return result
                else:
                    return "관련 내부 문서를 찾을 수 없습니다."
            else:
                return "내부 지식 문서(knowledge.txt)가 없습니다."
        except Exception as e:
            logger.error(f"내부 지식 검색 오류: {e}")
            return f"내부 문서 검색 중 오류가 발생했습니다: {str(e)}"
    
    try:
        # 유사도 검색
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return "관련 내부 문서를 찾을 수 없습니다."
        
        result = "📚 **내부 지식 문서 검색 결과:**\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"**[{i}]** {doc.page_content.strip()}\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"내부 지식 검색 오류: {e}")
        return f"내부 문서 검색 중 오류가 발생했습니다: {str(e)}"

# 도구 정의
tools = [
    Tool(
        name="keyword_extractor",
        description="사용자의 한국어 질문에서 특허 검색에 적합한 핵심 키워드를 추출합니다. 특허 검색 전에 반드시 이 도구를 먼저 사용하세요.",
        func=extract_search_keywords
    ),
    Tool(
        name="kipris_patent_searcher", 
        description="KIPRIS 특허 데이터베이스에서 특허를 검색합니다. keyword_extractor로부터 받은 JSON 결과를 입력으로 사용하세요.",
        func=search_kipris_patents
    ),
    Tool(
        name="internal_knowledge_searcher",
        description="회사 내부의 V-Belt 기술 문서와 제품 사양서를 검색합니다. 제품 추천이나 기술 사양 문의에 사용하세요.",
        func=search_internal_knowledge
    )
]

# LLM 및 에이전트 초기화 (LangChain 사용 가능한 경우에만)
if LANGCHAIN_AVAILABLE:
    # LLM 초기화 - Qwen3 모델 사용 (한국어 + 도구 호출 최적화)
    llm = ChatOllama(
        model="qwen3:8b",  # 한국어 + 도구 호출을 잘 지원하는 최신 모델
        temperature=0.1,
        num_predict=2048
    )

# 커스텀 에이전트 클래스 (도구 호출 미지원 모델용)
class SimpleAgent:
    """도구 호출을 지원하지 않는 모델을 위한 간단한 에이전트"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    
    def parse_and_execute_action(self, response_text: str) -> tuple[str, str]:
        """응답에서 도구 호출을 파싱하고 실행"""
        
        # 도구 호출 패턴 매칭
        import re
        
        # "도구명(인자)" 형태 매칭
        tool_pattern = r'(\w+)\s*\(\s*["\']([^"\']*)["\']?\s*\)'
        match = re.search(tool_pattern, response_text)
        
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name].func(tool_input)
                    return tool_name, result
                except Exception as e:
                    return tool_name, f"도구 실행 오류: {str(e)}"
        
        # 특허 검색 의도 감지
        if any(keyword in response_text for keyword in ['특허', '검색', 'KIPRIS']):
            if 'keyword_extractor' not in response_text:
                # 키워드 추출 먼저 실행
                keyword_result = self.tools['keyword_extractor'].func(response_text)
                patent_result = self.tools['kipris_patent_searcher'].func(keyword_result)
                return 'kipris_patent_searcher', patent_result
        
        # 내부 문서 검색 의도 감지  
        elif any(keyword in response_text for keyword in ['벨트', '모델', '사양', '제품']):
            result = self.tools['internal_knowledge_searcher'].func(response_text)
            return 'internal_knowledge_searcher', result
        
        return None, None
    
    def invoke(self, query_data: dict) -> dict:
        """에이전트 실행"""
        
        user_input = query_data.get("input", "")
        
        # 직접적인 도구 실행 방식
        try:
            # 특허 검색 요청 감지
            if any(keyword in user_input.lower() for keyword in ['특허', '검색', 'kipris']):
                logger.info("특허 검색 요청 감지")
                
                # 1단계: 키워드 추출
                keyword_result = self.tools['keyword_extractor'].func(user_input)
                logger.info(f"키워드 추출 결과: {keyword_result}")
                
                # 2단계: 특허 검색
                patent_result = self.tools['kipris_patent_searcher'].func(keyword_result)
                
                return {"output": patent_result}
            
            # 내부 문서 검색 요청 감지
            elif any(keyword in user_input.lower() for keyword in ['벨트', '모델', '사양', '제품', '추천']):
                logger.info("내부 문서 검색 요청 감지")
                
                result = self.tools['internal_knowledge_searcher'].func(user_input)
                return {"output": result}
            
            else:
                # 일반 응답
                return {"output": f"안녕하세요! 저는 전동벨트 및 컨베이어벨트 전문 어시스턴트입니다.\n\n다음과 같은 도움을 드릴 수 있습니다:\n• KIPRIS 특허 검색 (예: '타이밍벨트 특허 찾아줘')\n• 내부 제품 사양 및 기술 문서 검색 (예: '고온용 벨트 추천해줘')\n\n구체적인 질문을 해주세요."}
                
        except Exception as e:
            logger.error(f"에이전트 실행 오류: {e}")
            return {"output": f"처리 중 오류가 발생했습니다: {str(e)}"}

# LLM 및 에이전트 초기화 (LangChain 사용 가능한 경우에만)
if LANGCHAIN_AVAILABLE:
    # LLM 초기화 - 실제 사용 중인 한국어 모델로 설정
    llm = ChatOllama(
        model="my-korean-llm:latest",  # 사용자의 한국어 모델
        temperature=0.1,
        num_predict=2048
    )

    # 커스텀 에이전트 생성 (도구 호출 기능 없이)
    try:
        agent_executor = SimpleAgent(llm, tools)
        logger.info("커스텀 SimpleAgent가 성공적으로 생성되었습니다.")
    except Exception as e:
        logger.warning(f"커스텀 에이전트 생성 실패: {e}. 기본 응답 모드를 사용합니다.")
        agent_executor = None
else:
    agent_executor = None
    print("Warning: LangChain을 사용할 수 없어 기본 응답 모드로 실행됩니다.")

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        user_input = data.get("input", "")
        chat_history = data.get("chat_history", [])
        
        if not user_input.strip():
            return {"response": "질문을 입력해주세요."}
        
        logger.info(f"사용자 질문: {user_input}")
        
        # LangChain 에이전트 사용 가능한 경우
        if agent_executor and LANGCHAIN_AVAILABLE:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            response = result.get("output", "죄송합니다. 답변을 생성할 수 없습니다.")
        else:
            # 기본 응답 모드 (LangChain 없이)
            response = handle_basic_query(user_input)
        
        logger.info(f"응답 생성 완료")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"질의응답 처리 중 오류: {e}")
        return {"response": f"처리 중 오류가 발생했습니다: {str(e)}"}

def handle_basic_query(user_input: str) -> str:
    """LangChain 없이 기본적인 질문 처리"""
    
    try:
        # 특허 검색 의도 감지
        if any(keyword in user_input for keyword in ['특허', '검색', 'KIPRIS', 'kipris']):
            # 키워드 추출
            keyword_result = extract_search_keywords(user_input)
            # 특허 검색 실행
            search_result = search_kipris_patents(keyword_result)
            return search_result
        
        # 내부 지식 검색 의도 감지
        elif any(keyword in user_input for keyword in ['벨트', '모델', '사양', '제품', '추천']):
            search_result = search_internal_knowledge(user_input)
            return search_result
        
        else:
            return f"안녕하세요! 저는 전동벨트 및 컨베이어벨트 전문 어시스턴트입니다.\n\n다음과 같은 도움을 드릴 수 있습니다:\n• KIPRIS 특허 검색\n• 내부 제품 사양 및 기술 문서 검색\n\n구체적인 질문을 해주세요."
    
    except Exception as e:
        logger.error(f"기본 질의응답 처리 중 오류: {e}")
        return f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 전동벨트/컨베이어벨트 전문 AI 어시스턴트 시작")
    print("="*60)
    
    # 시스템 상태 체크
    if not LANGCHAIN_AVAILABLE:
        print("⚠️  LangChain이 설치되지 않았습니다.")
        print("고급 에이전트 기능을 위해 설치하세요:")
        print("pip install langchain langchain-ollama langchain-core")
        print("현재는 기본 응답 모드로 실행됩니다.")
    else:
        print("✅ Qwen3 모델이 설정되었습니다 (한국어 + 도구 호출 최적화).")
    
    if not KONLPY_AVAILABLE:
        print("⚠️  KoNLPy가 설치되지 않았습니다.")
        print("더 정확한 한국어 키워드 추출을 위해 설치하세요:")
        print("pip install konlpy")
        print("현재는 규칙 기반 키워드 추출을 사용합니다.")
    else:
        print("✅ KoNLPy 형태소 분석기가 활성화되었습니다.")
    
    if not VECTORSTORE_AVAILABLE:
        print("⚠️  벡터스토어를 사용할 수 없습니다.")
        print("고급 RAG 기능을 위해 설치하세요:")
        print("pip install langchain-huggingface sentence-transformers faiss-cpu")
        print("현재는 키워드 매칭 검색을 사용합니다.")
    else:
        print("✅ 벡터스토어 기반 RAG 시스템이 활성화되었습니다.")
    
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)