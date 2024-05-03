# # # 음성파일을 텍스트로 바꾸는 기능

# # # STEP 1
# # from transformers import pipeline


# # # STEP 2
# # transcriber = pipeline(task="automatic-speech-recognition")

# # # STEP 3
# # audio = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

# # # STEP 4
# # result = transcriber(audio)

# # # STEP 5
# # print(result)

# # STEP1
# from transformers import pipeline


# # STEP2
# vision_classifier = pipeline(model="google/vit-base-patch16-224")

# # STEP3
# image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

# # STEP4
# preds = vision_classifier(image_url)

# # STEP5
# preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
# print(preds)


# =============요약 AI
# text = "WASHINGTON, April 14 (Xinhua) -- U.S. President Joe Biden told Israeli Prime Minister Benjamin Netanyahu during a call on Saturday that the United States will oppose any Israeli counterattack against Iran, U.S. news portal Axios reported, citing a senior White House official."
# from transformers import pipeline

# summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
# result = summarizer(text)
# print(text)
# print("===========================")
# print(result)

# ===================
# text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."

# from transformers import pipeline

# translator = pipeline("translation", model="google-t5/t5-base")
# result = translator(text)
# print(result)

# ===============================================
# text = "all die"
# from transformers import pipeline

# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
# result=classifier(text)
# print(result)


from transformers import pipeline

question_answerer = pipeline("question-answering", model="FlagAlpha/Llama2-Chinese-7b-Chat")

question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

result = question_answerer(question=question, context=context)

print(result)