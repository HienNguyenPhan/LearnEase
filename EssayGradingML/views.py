from django.shortcuts import render
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai


model = load('./Models/model.joblib')
vectorizer = load('./Models/vectorizer.joblib')
model2 = genai.GenerativeModel("gemini-pro")
genai.configure(api_key="AIzaSyCKfnvX5UPPGIkxjmmn89r20jAliKq73kg")

# Create your views here.
def predictor(request):
    if request.method == 'POST':
        essay = request.POST['essay']
        new_essay_vectorized = vectorizer.transform([essay])
        y_pred = model.predict(new_essay_vectorized)
        y_pred[0] = round_to_nearest_half(y_pred[0])
        prompt = "Can you review the following essay based on these four catergories, Task Achievement, Coherence and Cohesion Lexical Resource, Grammatical Range and Accuracy, the review of each category should be short but precise and strict: " + essay
        result = model2.generate_content(prompt)
        response = result.text
        comment_items = response.split("**")
        comment_items = comment_items[1::]
        return render(request, 'main.html', {'result': y_pred[0], 'result2' : comment_items})
    return render(request, 'main.html')



def round_to_nearest_half(num):
    integer_part = int(num)
    decimal_part = num - integer_part

    if decimal_part <= 0.25:
        return integer_part
    elif decimal_part <= 0.75:
        return integer_part + 0.5
    else:
        return integer_part + 1