from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from zhipuai_llm import ZhipuAILLM
from langchain.vectorstores.chroma import Chroma
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import sys
from dotenv import load_dotenv, find_dotenv
import os
##界面
sys.path.append("/notebook/C3 搭建知识库") # 将父目录放入系统路径中
_ = load_dotenv(find_dotenv()) 





####################
def generate_response(input_text,zhipu_api_key):
    llm=ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = zhipu_api_key) 
    # 使用st.info来在蓝色框中显示AI生成的响应
    st.info(llm(input_text))

# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#     submitted = st.form_submit_button('Submit')
#     ##好像智谱的api没有特殊规律
#     if zhipu_api_key == '':
#         st.warning('Please enter your 智谱AI API key!', icon='⚠')
#     if submitted:
#         generate_response(text)


#添加检索问答
def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()

    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb
#带有历史记录的问答链

def get_chat_qa_chain(question:str,zhipu_api_key):
    vectordb = get_vectordb()
    llm=ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = zhipu_api_key) 
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']
#不带历史记录的问答链
def get_qa_chain(question:str,zhipu_api_key):
    vectordb = get_vectordb()
    llm=ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = zhipu_api_key) 
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]


##############
def main():
    st.title('大模型应用')
    zhipu_api_key = st.sidebar.text_input("zhipu API Key ",type='password') 
    
    selected_method = st.radio(
            "你想选择哪种模式进行对话？",
            ["None", "qa_chain", "chat_qa_chain"],
            captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])


    if 'messages' not in st.session_state:
        st.session_state.messages = []
    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})
        # 调用 respond 函数获取回答

        if selected_method == "None":
            answer = generate_response(prompt,zhipu_api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt,zhipu_api_key)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt,zhipu_api_key)


        
        # 检查回答是否为 None

        if answer is not None:
                # 将LLM的回答添加到对话历史中
                st.session_state.messages.append({"role": "assistant", "text": answer})
        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

if __name__ == "__main__":
    main()


