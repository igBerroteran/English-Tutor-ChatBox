from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv 
import requests
from playsound import playsound  
import os  
from flask import Flask, render_template, request
import chainlit as cl

huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']

load_dotenv(find_dotenv())

repo_id = "tiiuae/falcon-7b-instruct"
lm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.6, "max_new_tokens":2000})

template = """

    comportate como un profesor de ingles, manejar todos los niveles, desde A1 hasta C2 corrige mi pronunciación, mi gramatica, mi escritura, hablame de cultura inglesa, de literatura, corrigeme desde los niveles más faciles, a los niveles más basicos, tambien dama instruciones de fonetica, y puedes hacer comparaciones entre el ingles y el español
    para decir algo, por ejemplo, para decir I love you, se pronuncia hay lobeiu. No debes ser raro, no debes ser negativo, debes ser un poco entusiasta y no puedes ser aburrido.  
    {history}
    Alumno:{human_input}
    Jack:
    """


    
def factory(human_input):
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )
    llm_chain = LLMChain(
        llm=HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.2}),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )
    output = llm_chain.predict(human_input=human_input)

    return output


#web GUI
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message = factory(human_input)
    return message or ''

if __name__ == '__main__':
    app.run(debug=True)



