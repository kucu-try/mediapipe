
# # =============요약 AI
# # step1
from transformers import pipeline
from fastapi import FastAPI, Form


# # step2
# summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")



text = "all die"
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")





app = FastAPI()


@app.post("/analysis/")
async def login(text: str = Form()):  # step3

    # step4
    result=classifier(text)

    # step5
    return result