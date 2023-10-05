import streamlit as st
import pickle
import re
import nltk
from textblob import TextBlob

nltk.download('punkt')
nltk.download('stopwords')

# loding model
# This is a trained model using KNN algorithm
clf = pickle.load(open('clf.pkl','rb'))
# This model removes the stopwords and make sparse_matrix
tfidf = pickle.load(open('tfidf.pkl','rb'))

def cleanResume(txt):
    cleanTxt = re.sub('http\S+\s', ' ', txt)
    cleanTxt = re.sub('@\S+', ' ', cleanTxt )
    cleanTxt = re.sub('#\S+\s', ' ', cleanTxt )
    cleanTxt = re.sub('[%s]' % re.escape(
        """!"#$%'()*+,-./:;<>=_?@[\]^`{|}~"""), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)
    return cleanTxt


def analyze_sentiment(sparse_matrix):
    # Convert the sparse matrix back to text using inverse TF-IDF transform
    cleaned_resume = tfidf.inverse_transform(sparse_matrix)[0]
    cleaned_resume_text = ' '.join(cleaned_resume)

    if not cleaned_resume_text.strip():
        st.write("Error: No valid text for sentiment analysis.")
        return None

    analysis = TextBlob(cleaned_resume_text)
    sentiment_score = analysis.sentiment.polarity
    return sentiment_score

def main():
    st.title("Resume Anaylsis App")
    uploaded_file = st.file_uploader('Uplode Resume', type=['pdf','txt','docx'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8', errors='replace')

        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
            return

        cleaned_resume = cleanResume(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(cleaned_resume)[0]
        st.write(prediction_id)

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

        # Check if there are non-zero values in the sparse matrix
        if cleaned_resume.nnz > 0:
            sentiment_score = analyze_sentiment(cleaned_resume)
            # Map sentiment score to a percentage
            sentiment_percentage = (sentiment_score + 1) * 50

            sentiment_label = "Positive" if sentiment_score > 0 \
                else "Negative" if sentiment_score < 0 else "Neutral"

            st.write("Resume Score:", sentiment_score)
            st.write("Resume Score Percentage:", f"{sentiment_percentage:.2f}%")
            st.write("Sentiment Label:", sentiment_label)
        else:
            st.write("Error: No valid text for sentiment analysis.")





if __name__ == "__main__":
    main()