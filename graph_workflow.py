from typing import List, TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

from models import LLM_MODEL

# -----------------------------------------------------------------------
# LangGraph에서 상태(State)를 표현할 TypedDict 정의
# 각 노드 함수는 GraphState를 입력으로 받고, GraphState의 부분집합을 반환
# -----------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    question: str               # 사용자 질문 원문
    documents: List[str]        # 검색된 문서 리스트
    generation: str             # LLM이 생성한 답변 텍스트
    sub_queries: List[str]      # 질문 분해 결과로 생성된 하위 질문 리스트
    hallucination_status: str   # 환각 여부 평가 결과 ('yes' 또는 'no')
    hallucination_retries: int  # 환각 발생 시 재생성(re-generate) 시도 횟수


# --- 구조화된 출력 데이터 모델 ---
# 03-OutputParser/01-PydanticOutputParser.ipynb 참고 
# Pydantic의 BaseModel을 정의하고, LLM에 with_structured_output() 메서드로 JSON 스키마 강제 
# 변경점: binary_score 필드를 str 타입으로 변경 (원래는 bool 타입)
class GradeDocuments(BaseModel):
    # 검색된 문서의 관련성 평가 스키마
    binary_score: str = Field(
        description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'로 평가"
    )
    # LLM이 이 스키마에 맞게 JSON 형태로 출력 → LangChain이 파싱


class GradeHallucination(BaseModel):
    # 생성된 답변의 환각 여부 평가 스키마 (Groundedness Check)
    binary_score: str = Field(
        description="답변이 문서에 기반하여 사실적으로 작성되었으면 'yes', 아니면 'no'"
    )
    # 답변이 근거 문서(context)에 기반했는지를 이진(yes/no)으로 판단하는 지표


# ------------------------------------------------------------------------------
# RAG + 품질관리 워크플로우를 정의하는 LangGraph 앱 생성 함수
# retriever: 이미 build_retriever에서 구성된 최종 retriever (Ensemble + Reranker)
# ------------------------------------------------------------------------------
def create_rag_graph(retriever):
    """Retriever가 주입된 LangGraph 앱 생성"""

    # --- 노드 함수 정의 ---
    
    # 13-LangChain-Expression-Language/04-LCEL-Advanced.ipynb의 기본 체인 구성
    def query_decomposition_node(state: GraphState) -> GraphState:
        """
        - 입력: question
        - 출력: sub_queries (질문을 검색하기 좋은 n개의 하위 질문 리스트),
                hallucination_retries 초기화
        """
        
        question = state["question"]    # 현재 상태에서 사용자 질문 가져오기
        
        # LLM에게 "검색하기 좋은 개의 하위 질문"으로 나누도록 지시하는 프롬프트 템플릿*
        prompt = ChatPromptTemplate.from_template(
            "질문을 검색하기 좋은 n개의 한국어 하위 질문으로 분리해줘. "
            "결과는 줄바꿈으로 구분해.\n질문: {question}"
        )
        
        # prompt를 StrOutputParser을 통해 텍스트로 변환*
        chain = prompt | LLM_MODEL | StrOutputParser()
        
        # 실제 질문(q)을 넣어 LLM 실행*
        response = chain.invoke({"question": question})
        
        # LLM이 반환한 여러 줄의 텍스트를 줄 단위로 split → 공백 제거*
        sub_queries = [q.strip() for q in response.split("\n") if q.strip()]
        
        # 재시도 횟수 초기화
        return {"sub_queries": sub_queries,     # 분해된 하위 질문 리스트
                "hallucination_retries": 0      # 환각 재시도 카운터 초기화
                }

    # 10-Retriever
    def retrieval_node(state: GraphState) -> GraphState:
        """
        - 입력: sub_queries
        - 각 하위 질문에 대해 retriever.invoke(query)를 호출한 후
          모든 결과 문서 내용을 하나의 리스트로 모음
        - 출력: documents (중복 제거된 문자열 리스트)
        """
        sub_queries = state.get("sub_queries", [])  # 하위 질문 리스트 가져오기 (multi-query*)
        all_docs: List[str] = []                    # 모든 문서 내용을 담을 리스트 초기화 

        for q in sub_queries: #sub_queries의 각 하위 질문에 대해 반복*
            # build_retriever에서 만든 Ensemble Retriever 호출*
            docs = retriever.invoke(q)
            for d in docs:
                # 각 Document 객체에서 page_content만 추출하여 리스트에 추가
                all_docs.append(d.page_content)

        # 같은 내용의 문서가 중복될 수 있으므로 set으로 한번 중복 제거 후 다시 리스트로*
        unique_docs = list(set(all_docs))
        return {"documents": unique_docs}   # 상태에 documents 필드로 저장


    # 12-RAG
    def grade_documents_node(state: GraphState) -> GraphState:
        """
        - 입력: question, documents (문자열 리스트)
        - 각 문서가 질문과 관련 있는지 LLM으로 평가
        - 관련성이 "yes"인 문서만 필터링해서 반환
        """
        question = state["question"]
        documents = state.get("documents", [])
        
        # LLM을 Structured Output 모드로 사용 (GradeDocuments 스키마에 맞게 출력 강제)*
        structured_llm_grader = LLM_MODEL.with_structured_output(GradeDocuments)
        
        # LLM에게 평가자의 역할과 기준을 설명*
        system_msg = (
            "당신은 검색된 문서가 사용자의 질문과 관련이 있는지 평가하는 채점자입니다. "
            "문서에 질문과 관련된 키워드나 의미가 포함되어 있다면 'yes'로 평가하세요. 그렇지 않다면 'no'로 평가하세요."
            "엄격할 필요는 없습니다. 관련성이 조금이라도 있다면 'yes'를 주세요."
        )
        
        # 실제 질문과 단일 문서를 넣어 호출*
        grade_prompt = ChatPromptTemplate.from_messages(
            [("system", system_msg), ("human", "질문: {question}\n\n문서: {document}")]
        )
        
        # 프롬프트 + LLM(구조화 출력) 체인 구성
        retrieval_grader = grade_prompt | structured_llm_grader
        
        filtered_docs = []  # 관련성이 yes로 평가된 문서만 모을 리스트 초기화
        for doc in documents: 
             # 각 문서에 대해 LLM 평가 수행 *
            score = retrieval_grader.invoke({"question": question, "document": doc})
            # binary_score가 "yes"일 때만 문서 채택
            if score.binary_score.lower() == "yes":
                filtered_docs.append(doc)
        
        return {"documents": filtered_docs} # 상태에 필터링된 documents만 다시 저장


    # 12-RAG/03-Conversation-With-History.ipynb의 generate 체인 참고
    def generate_node(state: GraphState) -> GraphState:
        """
        - 입력: question, documents, hallucination_retries
        - 문서들을 하나의 context로 합쳐 LLM에 전달하고,
        질문 유형에 따라 학술적/전문적 답변 제공
        - 환각 재시도가 있는 경우, 더 엄격하게 문서 기반으로 답변하라는 guidance 추가
        - 출력: generation (최종 답변 텍스트)
        """
        question = state["question"]
        documents = state.get("documents", [])
        context = "\n\n".join(documents)
        retries = state.get("hallucination_retries", 0)
        
        # 질문 유형 감지 및 적응형 guidance 생성
        guidance = _generate_adaptive_guidance(question, retries)

        # 질문 유형에 따른 적응형 프롬프트 템플릿
        prompt = ChatPromptTemplate.from_template(
            "당신은 논문 분석 전문가입니다. 다음 규칙을 반드시 지켜주세요.\n"
            "아래 문서를 바탕으로 질문에 대해 학술적으로 답변해줘.\n"
            "구체적인 알고리즘명이나 숫자는 정확하게 기억해줘.\n"
            "모든 주장은 제시된 문서의 근거로 뒷받침되어야 합니다.\n"
            "문서에 없는 내용이라면 지어내지 말고 '문서에는 해당 정보가 없습니다'라고 명시하세요.\n"
            "{guidance}\n\n"
            "[문서]\n{context}\n\n[질문]\n{question}"
        )
        
        chain = prompt | LLM_MODEL | StrOutputParser()
        generation = chain.invoke({"context": context, "question": question, "guidance": guidance})
        
        return {"generation": generation}


    def _generate_adaptive_guidance(question: str, retries: int) -> str:
        """
        질문 유형을 분석하여 적응형 가이드 메시지 생성
        
        질문 유형:
        1. 정의/개념 질문 (what/정의)
        2. 비교/대조 질문 (vs/차이/비교)
        3. 원인/결과 질문 (why/cause/영향)
        4. 방법론 질문 (how/방법/프로세스)
        5. 평가/분석 질문 (평가/분석/장단점)
        """
        
        question_lower = question.lower()
        
        # 재시도 시 공통 기본 가이드
        base_guidance = ""
        if retries > 0:
            base_guidance = "⚠️ 이전 답변이 부정확했습니다. 반드시 문서의 내용만 사용하여 정확히 다시 작성하세요. 문서에 명시되지 않은 해석이나 추론은 피하세요.\n"
        
        # 질문 유형별 가이드
        if any(keyword in question_lower for keyword in ["정의", "뜻", "무엇", "what is", "definition"]):
            # 정의/개념 질문
            return (
                base_guidance +
                "📌 답변 형식:\n"
                "1) 핵심 정의를 명확하고 간결하게 제시\n"
                "2) 핵심 특성 또는 요소 (최대 3개) 열거\n"
                "3) 학문 분야에서 해당 개념의 위치 또는 중요성 설명\n"
                "4) 실제 사례나 적용 분야 (문서에 있는 경우만)"
            )
        
        elif any(keyword in question_lower for keyword in ["비교", "vs", "차이", "다르", "차이점", "difference", "compare", "contrast"]):
            # 비교/대조 질문
            return (
                base_guidance +
                "📊 답변 형식:\n"
                "1) 비교 대상의 정의 간단히 제시\n"
                "2) 표 형식 또는 항목별로 주요 차이점 명시 (예: 정의, 특성, 용도, 장점, 한계 등)\n"
                "3) 각 차이가 의미하는 바를 학술적으로 해석\n"
                "4) 어떤 상황에서 어느 것을 사용하는지 (문서 기반)"
            )
        
        elif any(keyword in question_lower for keyword in ["왜", "원인", "이유", "why", "cause", "reason", "영향", "effect", "impact"]):
            # 원인/결과/영향 질문
            return (
                base_guidance +
                "🔗 답변 형식:\n"
                "1) 인과관계를 명확하게 설명\n"
                "2) 원인 → 메커니즘 → 결과의 단계별 논리 전개\n"
                "3) 중간 단계의 세부 메커니즘 설명 (문서에서 지원되는 경우)\n"
                "4) 예외 상황이나 조건부 결과 언급 (있다면)\n"
                "5) 관련 연구나 이론적 배경 (문서에 있다면)"
            )
        
        elif any(keyword in question_lower for keyword in ["어떻게", "방법", "프로세스", "절차", "단계", "how", "process", "procedure", "step"]):
            # 방법론/프로세스 질문
            return (
                base_guidance +
                "🔧 답변 형식:\n"
                "1) 전체 프로세스의 개요 제시\n"
                "2) 각 단계를 순서대로 명확하게 설명 (단계별 제목 포함)\n"
                "3) 각 단계별 주의점, 조건, 또는 필수 요소\n"
                "4) 단계 간의 연계성과 의존성\n"
                "5) 결과 검증 방법 또는 성공 기준 (있다면)"
            )
        
        elif any(keyword in question_lower for keyword in ["평가", "분석", "장점", "단점", "장단점", "평가", "assess", "analysis", "advantage", "disadvantage", "strength", "weakness"]):
            # 평가/분석/장단점 질문
            return (
                base_guidance +
                "⚖️ 답변 형식:\n"
                "1) 평가 기준 또는 분석 틀 명시\n"
                "2) 긍정적 측면 (근거 제시)\n"
                "3) 부정적 측면 또는 한계 (근거 제시)\n"
                "4) 중립적 관점에서 상황별 적용 가능성\n"
                "5) 학술적 평가 또는 현장 전문가의 견해 (문서 기반)"
            )
        
        else:
            # 기타 일반적인 질문
            return (
                base_guidance +
                "📝 답변 형식:\n"
                "1) 질문의 핵심 개념을 명확히 정의\n"
                "2) 중요 내용을 논리적 순서대로 전개\n"
                "3) 각 주장은 문서의 구체적 근거로 뒷받침\n"
                "4) 필요시 도표, 예시, 또는 사례 포함\n"
                "5) 결론은 명확하고 재현 가능한 표현으로 마무리"
            )

    # 12-RAG
    def web_search_node(state: GraphState) -> GraphState:
        """
        - 입력: question
        - Tavily 검색 도구를 사용해 웹에서 k=3개의 결과를 가져옴
        - 각 결과의 content만 추출하여 documents 리스트로 반환
        - 예외 발생 시, 에러 메시지를 documents로 반환
        """
        try:
            tool = TavilySearchResults(k=3)                     # Tavily 검색 도구 인스턴스 생성
            docs = tool.invoke({"query": state["question"]})    # 웹 검색 실행
            web_content = [d["content"] for d in docs]          # 결과에서 content만 추출
            return {"documents": web_content}
        except Exception:
            return {"documents": ["웹 검색 도구를 사용할 수 없습니다."]}    # 검색 실패 시에도 그래프가 죽지 않고 fallback 텍스트를 반환
    
    
    # 12-RAG / 16-Evaluations
    def hallucination_check_node(state: GraphState) -> GraphState:
        """
        - 입력: documents (근거 문서), generation (LLM 답변),
                hallucination_retries
        - LLM을 사용해 답변이 문서에 근거했는지 평가
        - 환각이면 retries를 1 증가
        - 출력: hallucination_status ('yes' or 'no'), hallucination_retries
        """
        documents = state.get("documents", [])
        generation = state["generation"]        # generate_node에서 생성된 답변
        context = "\n\n".join(documents)        # 근거로 사용할 문서 context
        
        # 현재까지의 재시도 횟수
        retries = state.get("hallucination_retries", 0)

         # 환각 판별용 LLM을 구조화 출력 모드로 사용*
        structured_llm_grader = LLM_MODEL.with_structured_output(GradeHallucination)
        system_msg = (
            "당신은 AI 답변이 참조 문서(Context)에 기반했는지 검증하는 채점자입니다. "
            "답변이 문서의 내용으로 뒷받침된다면 'yes', 문서에 없는 내용을 지어냈다면 'no'로 평가하세요."
            "만약 문서에 정보가 없어서 '모른다'고 답한 경우에도 'yes'(사실 기반)로 평가하세요."
        )
        
        # context(문서들)와 generation(답변)을 함께 보여주고 판단을 요청하는 프롬프트*
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("human", "문서: {context}\n\n답변: {generation}")
            ]
        )
        hallucination_grader = grade_prompt | structured_llm_grader
        
        # LLM에게 context와 generation을 넘겨 환각 여부 판단 실행*
        score = hallucination_grader.invoke({"context": context, "generation": generation})
        
        status = score.binary_score.lower() # "yes" 또는 "no"*
        
        # 환각으로 판정된 경우 재시도 횟수 증가*
        if status == "no":
            retries += 1
            
        return {"hallucination_status": status, "hallucination_retries": retries}   # 상태에 환각 여부와 최신 재시도 카운터를 저장


    # -------------------------------------------------------------------
    #                       라우팅 함수들 (조건 분기 로직)
    # -------------------------------------------------------------------
    def decide_retrieval_route(state: GraphState) -> str:
        """
        grade 노드 이후의 분기 결정
        - documents가 비어 있으면: web_search 노드로*
        - documents가 있으면: generate 노드로*
        """
        documents = state.get("documents", [])
        if not documents:
            return "web_search"
        return "generate"

    def decide_hallucination_route(state: GraphState) -> str:
        """
        hallucination_check 노드 이후의 분기 결정
        1) status == 'yes' (환각 아님)          -> END
        2) status == 'no' & retries > 3         -> END (무한 루프 방지)
        3) status == 'no' & retries <= 3        -> generate (재시도)
        """
        status = state.get("hallucination_status", "yes")
        retries = state.get("hallucination_retries", 0)
        
        if status == "yes":
            return "end"    # END로 가는 라벨
        elif retries > 3:
            return "end"    # 최대 재시도 횟수를 초과하면 강제 종료 (또는 web_search로 보내는 것도 가능)
        else:
            return "generate"   # 환각이지만 아직 재시도 가능 → 다시 generate 노드로 루프


    # -------------------------------------------------------------------
    #                              그래프 조립
    # -------------------------------------------------------------------
    # GraphState 타입을 상태로 사용하는 StateGraph 인스턴스 생성
    # 17-LangGraph
    workflow = StateGraph(GraphState)
    
    # 각 노드를 그래프에 등록 (노드 이름, 함수)
    workflow.add_node("decompose", query_decomposition_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # 기본 엣지 구성: START → decompose → retrieve → grade
    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "grade")
    
    # grade 이후 분기:
    # - decide_retrieval_route(state)가 "web_search"를 반환하면 web_search 노드로
    # - "generate"를 반환하면 generate 노드로
    workflow.add_conditional_edges(
        "grade",
        decide_retrieval_route,
        {"web_search": "web_search", "generate": "generate"},
    )
    
    # web_search 이후에는 무조건 generate 노드로 이동
    workflow.add_edge("web_search", "generate")
    
    # generate 이후에는 hallucination_check 노드로 이동
    workflow.add_edge("generate", "hallucination_check")
    
    # hallucination_check 이후 분기:
    # - decide_hallucination_route(state)가 "end" → END
    # - "generate" → 다시 generate 노드로 루프
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_hallucination_route,
        {
            "end": END,
            "generate": "generate"
        }
    )

    return workflow.compile()