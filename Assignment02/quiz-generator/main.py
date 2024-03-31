from flask import Flask, render_template
from flask import request
from flask import Response                                
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader, PyPDFDirectoryLoader, PyPDFLoader, DirectoryLoader
from langchain.vectorstores import DocArrayInMemorySearch,FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import SentenceTransformerEmbeddings, VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.docstore.document import Document
import os 
import json
import time
from PyPDF2 import PdfFileReader
import csv
import pandas as pd
import vertexai                                           #<-CHANGED
from vertexai.language_models import TextGenerationModel  #<-CHANGED


app = Flask(__name__, template_folder="templates")  # Create a Flask object.
PORT = os.environ.get("PORT")  # Get PORT setting from environment.
if not PORT:
    PORT = 8080

TOPIC_NO = str([i+1 for i in range(len(os.listdir('data/')))])
print(TOPIC_NO)

# Initialize Vertex AI access.
vertexai.init(project="cloudquizgenerator002", location="us-central1")  #<-CHANGED
parameters = {                                                 #<-CHANGED
    "candidate_count": 1,                                      #<-CHANGED
    "max_output_tokens": 1024,                                 #<-CHANGED
    "temperature": 0.5,                                        #<-CHANGED
    "top_p": 0.9,                                              #<-CHANGED
    "top_k": 40,                                               #<-CHANGED
}                                                              #<-CHANGED
model = TextGenerationModel.from_pretrained("text-bison@001")             #<-CHANGED

########################################################################
def summarizeContent(text):
    PROMPT = "Summarize the following content: " + text + ". Don't miss out any important information"
    response = model.predict(PROMPT, **parameters,)
    return response.text

# convert pdf files to text
def PDFLoaders(path):
    loader = PyPDFLoader(path)
    # Load data from PDF
    data = loader.load()

    question_gen = ''
    summarizedContent = ''
    count = 0
    for page in data:
        question_gen += page.page_content
        count += 1
        if count %10 == 0:
            summarizedContent += summarizeContent(question_gen)
            question_gen = ''


    splitter_ques_gen = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=10000, chunk_overlap=1000)

    chunks_ques_gen = splitter_ques_gen.split_text(summarizedContent)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=20)


    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen

def buildQA(filename):
    path = 'data/' + filename
    document_ques_gen, document_answer_gen = PDFLoaders(path)
    llm_ques_gen_pipeline = VertexAI(model_name="text-bison@001",
                       project='cloudquizgenerator002',
                       temperature=0.3)

     
    prompt_template = """
    You are an expert at creating questions based on coding materials and documentation.
    Your goal is to prepare a coder or programmer for their exam and coding tests.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create questions that will prepare the coders or programmers for their tests.
    Make sure not to lose any important information. DO NOT NUMBER THE QUESTIONS.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    refine_template = ("""
    You are an expert at creating practice questions based on coding material and documentation.
    Your goal is to help a coder or programmer prepare for a coding test.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions. DO NOT NUMBER THE QUESTIONS!
    QUESTIONS:
    """
    )
    
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                            chain_type = "refine",
                                            question_prompt=PROMPT_QUESTIONS,
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = VertexAIEmbeddings()

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = VertexAI(model_name = "text-bison@001", project = "cloudquizgenerator002", temperature = 0.1)

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                chain_type="stuff",
                                                verbose = True,
                                                retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list
               
def get_csv(topic):
    base_folder = 'output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder + "QA.csv"
    topic = topic.strip('][').split(', ')
    topicIndex = [int(i)-1 for i in topic]
    listPDF = sorted([fn for fn in os.listdir('data/') if fn.endswith('.pdf')])
    listPDF_wanted = [listPDF[x] for x in topicIndex]
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer", "Filename"])  # Writing the header row
        for path in listPDF_wanted:
            answer_generation_chain, ques_list = buildQA(path)
            print("Filename: ", path)
            for question in ques_list:
                print("Question: ", question)
                answer = answer_generation_chain.run(question)
                print("Answer: ", answer)
                print("--------------------------------------------------\n\n")
                # Save answer to CSV file
                csv_writer.writerow([question, answer, path])
    return output_file     

########################################################################
# This function takes a dictionary, a name, and a default value.
# If the name exists as a key in the dictionary, the corresponding
# value is returned. Otherwise, the default value is returned.
def check(args, name, default):
    if name in args:
        return args[name]
    return default
    
# The app.route decorator routes any GET requests sent to the /generate
# path to this function, which responds with "Generating:" followed by
# the body of the request.
@app.route("/", methods=["GET"])
# This function generates a quiz using Vertex AI.
def generateLargeChunks():
    args = request.args.to_dict() 
    topic = check(args, "topicNo", TOPIC_NO)
    output_file = get_csv(topic)
    df = pd.read_csv(output_file)
    dtHTML = df.to_html(index = False)
    return render_template('index.html', table=dtHTML)
    
    # topic = topic.strip('][').split(', ')
    # topicIndex = [int(i)-1 for i in topic]
    # print(sorted([fn for fn in os.listdir('data/') if fn.endswith('.pdf')]))
    # listPDF = sorted([fn for fn in os.listdir('data/') if fn.endswith('.pdf')])
    # listPDF_wanted = [listPDF[x] for x in topicIndex]
    # print(listPDF_wanted)
    # def generate():
    #     for path in listPDF_wanted:
    #         answer_generation_chain, ques_list = buildQA(path)
    #         for question in ques_list:
    #             print("Question: ", question)                
    #             answer = answer_generation_chain.run(question)
    #             print("Answer: ", answer)
    #             print("--------------------------------------------------\n\n")
    #             yield f"Question: {question}\nAnswer: {answer}\nSource: {path}"
    # return app.response_class(generate())
        
        


# This code ensures that your Flask app is started and listens for
# incoming connections on the local interface and port 8080.
if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=PORT, threaded = True)