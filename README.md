![image](https://github.com/bobpeulen/gen_ai/assets/75362416/c4eb20ce-7c87-4b6c-a675-ce3075baa24b)# gen_ai
All GenAI, LangChain, Vector db related topics


- gen_ai_langchain_faiss.ipynb -> A small Gradio application that includes:
   - Uploading PDF file(s)
   - Chunck the PDF is pieces and create embeddings from the texts
   - Store embeddings in FAISS
   - Invoke GenAI using LangChain with the questions
   - Use history to ask follow-up questions
  
- demo-RAG-deployment.ipynb. Qdrant vector db + Q&A + OCI GenAI + Gradio (No LangChain)
- small_genai.py. Python SDK for OCI GenAI
- speech_to_text_and_genai.py
- client_handbags.ipynb. Images of handbags converted to embeddings, stored in vector db, queried by sending new image and returning top 3 similar images.

- fashion_indentification.ipynb --> Using Yolov5 detecting clothing, deployed on OCI Data Science - Model Deployment. See artifact files
