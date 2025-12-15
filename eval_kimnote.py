import os
import getpass
from typing import List, Optional
from pydantic import BaseModel, Field

# LangSmith 관련 임포트
from langsmith import Client, evaluate

# LangChain 관련 임포트
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Kimnote 프로젝트 모듈 임포트
from retriever_builder import build_retriever
from graph_workflow import create_rag_graph

# -----------------------------------------------------------------------------
# 1. 환경 설정 및 데이터셋 준비
# -----------------------------------------------------------------------------

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
if "LANGCHAIN_API_KEY" not in os.environ:
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "KW-RAG"

# 평가용 LLM (채점자)
# 코드가 이미 검증된 ChatOpenAI를 사용합니다.
eval_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 평가 데이터셋
EVAL_DATASET = [
    {
        "inputs": {"question": "이 논문의 주요 공헌(contributions)은 무엇인가요?"},
        "outputs": {"answer": "재밍 환경에서의 UAV-기반 Semantic Communication–MEC 통합 프레임워크 제안, 혼합 연속–이산 의사결정을 위한 T5D(DT3+DDQN) 기반 DRL 알고리즘 설계, 지능형 재머를 포함한 적대적 학습 구조 모델링, Semantic Communication 관점에서의 성능 지표 및 시스템 분석 입니다. "}
    },
    {
        "inputs": {"question": "논문에서 시스템의 최적화를 위해 사용한 두 가지 핵심 딥 강화학습(DRL) 알고리즘은 무엇입니까?"},
        "outputs": {"answer": "Deep Q-Learning (DQL) 알고리즘과 Dueling Deep Q-Learning (Dueling DQL) 알고리즘입니다. "}
    },
    {
        "inputs": {"question": "시뮬레이션 결과에서 Dueling DQL이 기존 DQL 방식보다 더 나은 성능을 보이는 이유는 무엇인가요?"},
        "outputs": {"answer": "Dueling DQL은 상태 가치(Value function)와 행동 이점(Advantage function)을 분리하여 추정하는 구조를 가집니다. 이로 인해 모든 상태-행동 쌍을 탐색하지 않아도 학습이 가능하여 수렴 속도가 더 빠르고, 더 안정적인 보상(reward)을 얻을 수 있기 때문입니다."}
    }
]

# -----------------------------------------------------------------------------
# 2. Kimnote RAG 시스템 래핑 (Target Function)
# -----------------------------------------------------------------------------

def initialize_kimnote_app(pdf_path: str):
    print(f"Loading retriever from: {pdf_path}")
    retriever = build_retriever(pdf_path)
    if not retriever:
        raise ValueError("Retriever 생성 실패. PDF 경로를 확인하세요.")
    app = create_rag_graph(retriever)
    return app

# ★ 주의: 실제 평가할 PDF 파일 경로로 수정해주세요 ★
PDF_PATH = "./data/sample_paper.pdf" 
rag_app = None

if os.path.exists(PDF_PATH):
    rag_app = initialize_kimnote_app(PDF_PATH)
else:
    print(f"⚠️ 경고: '{PDF_PATH}' 파일을 찾을 수 없습니다.")

def predict_kimnote(inputs: dict) -> dict:
    if rag_app is None:
        return {"output": "앱이 초기화되지 않았습니다.", "contexts": []}

    question = inputs["question"]
    response_state = rag_app.invoke({"question": question})
    
    final_answer = response_state.get("generation", "")
    retrieved_docs = response_state.get("documents", [])
    
    return {
        "output": final_answer,
        "contexts": retrieved_docs
    }

# -----------------------------------------------------------------------------
# 3. 직접 구현한 평가 로직 (Custom Evaluators)
# -----------------------------------------------------------------------------

# 3.1 답변 정확성 평가 (QA Correctness)
class CorrectnessScore(BaseModel):
    score: int = Field(description="답변의 정확성 점수 (1: 부정확 ~ 5: 매우 정확)")
    reasoning: str = Field(description="점수 부여 이유")

def evaluate_correctness(run, example):
    """
    정답(Ground Truth)과 예측(Prediction)을 비교하여 정확성을 1~5점으로 평가
    """
    prediction = run.outputs["output"]
    reference = example.outputs["answer"]
    input_question = example.inputs["question"]

    # 채점용 프롬프트
    prompt = ChatPromptTemplate.from_template(
        """당신은 공정한 채점관입니다. 아래 질문에 대한 '실제 정답(Ground Truth)'과 AI가 생성한 '예측 답변(Prediction)'을 비교하여 평가해주세요.

[질문]: {question}
[실제 정답]: {reference}
[예측 답변]: {prediction}

예측 답변이 실제 정답의 핵심 의미를 잘 포함하고 있는지 판단하여 1점에서 5점 사이의 점수를 부여하세요.
1점: 완전히 틀림
3점: 일부 맞으나 누락되거나 부정확한 내용 있음
5점: 핵심 내용을 정확하게 포함함
"""
    )
    
    # 구조화된 출력으로 점수 추출
    evaluator = prompt | eval_llm.with_structured_output(CorrectnessScore)
    result = evaluator.invoke({
        "question": input_question,
        "reference": reference,
        "prediction": prediction
    })

    return {
        "key": "correctness",
        "score": result.score / 5.0,  # 0~1 스케일로 정규화
        "comment": result.reasoning
    }


# 3.2 문맥 기반 사실성 평가 (Groundedness / Hallucination Check)
class GroundednessScore(BaseModel):
    is_grounded: str = Field(description="답변이 문맥에 기반했는지 여부 ('yes' or 'no')")
    reasoning: str = Field(description="판단 이유")

def evaluate_groundedness(run, example):
    """
    답변이 검색된 문서(Contexts)에 기반했는지(환각 여부) 평가
    """
    prediction = run.outputs["output"]
    contexts = run.outputs["contexts"]
    input_question = example.inputs["question"]
    
    # 리스트 형태의 문서를 하나의 텍스트로 결합
    context_str = "\n\n".join(contexts) if isinstance(contexts, list) else str(contexts)

    # 채점용 프롬프트
    prompt = ChatPromptTemplate.from_template(
        """당신은 '환각(Hallucination)'을 탐지하는 검사관입니다.
AI의 답변이 제공된 '참조 문서(Context)'에 있는 내용만을 기반으로 작성되었는지 판단하세요.

[참조 문서]:
{context}

[AI 답변]:
{prediction}

답변의 모든 내용이 문서에 의해 뒷받침된다면 'yes', 문서에 없는 내용을 지어냈다면 'no'라고 답하세요.
"""
    )

    evaluator = prompt | eval_llm.with_structured_output(GroundednessScore)
    result = evaluator.invoke({
        "context": context_str,
        "prediction": prediction
    })
    
    # yes = 1점 (환각 없음), no = 0점 (환각)
    score = 1 if result.is_grounded.lower() == "yes" else 0

    return {
        "key": "groundedness",
        "score": score,
        "comment": result.reasoning
    }

# -----------------------------------------------------------------------------
# 4. LangSmith Dataset 생성 및 평가 실행
# -----------------------------------------------------------------------------

def run_evaluation():
    client = Client()
    dataset_name = "KimNote_Evaluation_Dataset_V1"

    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[e["inputs"] for e in EVAL_DATASET],
            outputs=[e["outputs"] for e in EVAL_DATASET],
            dataset_id=dataset.id,
        )
        print(f"✅ 데이터셋 '{dataset_name}' 생성 완료.")
    else:
        print(f"ℹ️ 기존 데이터셋 '{dataset_name}'을 사용합니다.")

    print("🚀 평가를 시작합니다... (LangSmith 대시보드를 확인하세요)")
    
    results = evaluate(
        predict_kimnote,
        data=dataset_name,
        evaluators=[
            evaluate_correctness,
            evaluate_groundedness
        ],
        experiment_prefix="kimnote-custom-eval",
        metadata={"description": "KimNote RAG Evaluation with Custom LLM Judges"}
    )
    
    print("\n🏁 평가 완료!")
    print(results)

if __name__ == "__main__":
    if rag_app:
        run_evaluation()
    else:
        print("❌ 실행 실패: PDF 경로를 확인하고 파일을 넣어주세요.")