import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def jaccard_similarity(set1, set2):
    """Compute Jaccard Similarity between two sets."""
    intersection_len = len(set1.intersection(set2))
    union_len = len(set1)
    return intersection_len / union_len if union_len != 0 else 0
def process_day_group(group):
    # Split the advice field into words
    texts = group['advice'].astype(str).str.split().tolist()
    max_length = len(texts)
    # print(max_length)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    best_coherence = -1
    best_perplexity = float('inf')
    best_topic_num = 0
    best_lda_model = None

    # Prepare lists to store coherence and perplexity values for visualization
    coherences = []
    perplexities = []

    # Iterate over a range of topic numbers to find the optimal number
    topic_range = range(2, 31)  # Assuming we check from 2 to 30 topics
    for num_topics in topic_range:
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        perplexity = lda_model.log_perplexity(corpus)

        # Store coherence and perplexity for visualization
        coherences.append(coherence)
        perplexities.append(perplexity)

        # Update best model based on coherence and perplexity
        if coherence > best_coherence and perplexity < best_perplexity:
            best_coherence = coherence
            best_perplexity = perplexity
            best_topic_num = num_topics
            best_lda_model = lda_model

    # Store best topics data
    topics_data = []
    for i in range(best_topic_num):
        for word, weight in best_lda_model.show_topic(i, topn=max_length):
            topics_data.append({
                "Topic": f"Topic_{i + 1}",
                "Word": word,
                "Weight": weight
            })
    #
    # Randomly select an admission_number1
    random_admission = np.random.choice(group['admission_number1'].unique())
    print(random_admission)
    # Get all the advice under the admission_number1 and merge it into one text
    treatment_process = " ".join(group[group['admission_number1'] == random_admission]['advice'].tolist())
    # Calculate the "documents" composed of keywords for each topic
    topics_keywords = [" ".join([word[0] for word in topics_data[topic]]) for topic in topics_data]

    # Calculate the Jaccard similarity between treatment_process and each subject
    treatment_words = set(treatment_process.split())
    jaccard_similarities = [jaccard_similarity(treatment_words, set(topic.split())) for topic in topics_keywords]
    return {
        "day": group['advice_day'].iloc[0],
        "disease_name": '心肌梗死',  # Add disease name to the results
        "topics_data": topics_data,
        "coherence": best_coherence,
        "perplexity": best_perplexity,
        "coherences": coherences,
        "perplexities": perplexities,
        "jaccard_similarity_scores": jaccard_similarities  # Added Jaccard scores to the results

    }

def main():
    data = pd.read_csv('I21pre_ready.csv')
    # '非ST抬高性心肌梗死'-NSTEMI,'心肌梗死' except '非ST抬高性心肌梗死'-STEMI
    filtered_data = data[(data['hospitalization_time'] >= 3) & (data['hospitalization_time'] <= 15) & (
        data['discharge_diagnosis'].str.startswith('心肌梗死'))]

    max_advice_day = filtered_data['advice_day'].max()

    topic_nums = list(range(2, 31))  # Assuming we check from 2 to 20 topics
    all_results = []
    for _, group in filtered_data.groupby('advice_day'):
        result = process_day_group(group)
        all_results.append(result)

        df_list = []
        for result in all_results:
            day = result['day']
            disease_name = result['disease_name']
            for topic_data in result['topics_data']:
                row_data = {
                    "Disease_Name": disease_name,  # Ensure Disease_Name is the first column
                    "Day": day,
                    "Topic": topic_data["Topic"],
                    "Word": topic_data["Word"],
                    "Weight": topic_data["Weight"]
                }
                df_list.append(row_data)

        df = pd.DataFrame(df_list)
        df.to_csv('feature_evaluation/daily_topics_心肌梗死.csv', index=False)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

