from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import answer_examples
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda
from operator import itemgetter  # 추가

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_llm():
    llm = ChatOpenAI(model='gpt-4o')
    return llm


def get_dictionary_chain():
    llm = get_llm()
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}

        질문: {{input}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()

    return dictionary_chain


def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    
    # 간소화된 RAG 체인
    system_prompt = """당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
    "아래에 제공된 문서를 활용해서 답변해주시고"
    "답변을 알 수 없다면 모른다고 답변해주세요"
    "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
    "2-3 문장정도의 짧은 내용의 답변을 원합니다"

Context: {context}

Chat History: {chat_history}

Question: {input}"""

    # 최종 프롬프트: System + Few-shot Examples + Context + Question
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,  # Few-shot 예제 추가
            ("system", "참고할 문서:\n{context}"),
            ("system", "대화 기록:\n{chat_history}"),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = get_retriever()
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,  # "input" 키에서 값 추출
            "chat_history": lambda x: x.get("chat_history", ""),
            "input": itemgetter("input")  # "input" 키에서 값 추출
        }
        | final_prompt
        | llm
        | StrOutputParser()
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_message_key="input",
        history_messages_key="chat_history",
        output_message_key="answer",
    )
    return conversational_rag_chain

def get_ai_response(user_message):
    load_dotenv()
    
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    
    # Step 1: 질문 변환
    reformed_question = dictionary_chain.invoke({"input": user_message})
    print(f"🔄 변환된 질문: {reformed_question}")
    
    # Step 2: RAG 실행 (스트리밍)
    print(f"🤖 AI 답변 생성 중...")
    ai_response = rag_chain.stream(
        {"input": reformed_question},
        config={"configurable": {"session_id": "abc123"}}
    )
    
    return ai_response