"""
LDA Topic Modeling for AI Harm Discourse Analysis
This script performs Latent Dirichlet Allocation (LDA) topic modeling
on Reddit posts about AI harm discourse.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)


# 1. TEXT PREPROCESSING

def preprocess_text(text, stop_words, lemmatizer):
    """
    Clean text for topic modeling
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
    stop_words : set
        Set of stopwords to remove
    lemmatizer : WordNetLemmatizer
        NLTK lemmatizer instance
        
    Returns:
    --------
    list : List of preprocessed tokens
    """
    if pd.isna(text) or text == '':
        return []
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove Reddit-specific patterns
    text = re.sub(r'/r/\w+', '', text)
    text = re.sub(r'u/\w+', '', text)
    
    # Remove special characters, keep only letters
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and short words
    tokens = [word for word in tokens 
              if word not in stop_words and len(word) > 3]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens


def setup_preprocessing():
    """
    Initialize preprocessing components
    
    Returns:
    --------
    tuple : (stop_words set, lemmatizer instance)
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords (domain-specific)
    custom_stops = {
        'ai', 'algorithm', 'algorithms', 'artificial', 'intelligence',
        'would', 'could', 'should', 'may', 'might',
        'using', 'used', 'use', 'one', 'two', 'three',
        'also', 'said', 'says', 'get', 'got', 'go',
        'https', 'http', 'www', 'com', 'reddit',
        'edit', 'amp', 'quot', 'nbsp', 'removed', 'deleted'
    }
    stop_words.update(custom_stops)
    
    return stop_words, lemmatizer

# 2. SKLEARN-BASED LDA (TF-IDF)

def train_lda_sklearn(df, text_col='processed_text', n_topics=30):
    """
    Train LDA model using sklearn with TF-IDF vectorization
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with processed text
    text_col : str
        Name of text column
    n_topics : int
        Number of topics to extract
        
    Returns:
    --------
    tuple : (topic_assignments, topic_words_dict, lda_model, vectorizer, dtm)
    """
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=20,
        max_df=0.5,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Document-term matrix
    dtm = vectorizer.fit_transform(df[text_col])
    
    # LDA Model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='batch',
        max_iter=50
    )
    
    # Fit model
    lda.fit(dtm)
    
    # Get topic assignments
    lda_output = lda.transform(dtm)
    topic_assignments = lda_output.argmax(axis=1)
    
    # Extract topic words
    feature_names = vectorizer.get_feature_names_out()
    topic_words = {}
    
    for topic_idx in range(n_topics):
        top_indices = lda.components_[topic_idx].argsort()[-20:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_words[topic_idx] = top_words
        
        print(f"\nLDA Topic {topic_idx}:")
        print(f"  {', '.join(top_words[:10])}")
    
    return topic_assignments, topic_words, lda, vectorizer, dtm


# 3. GENSIM-BASED LDA (with Coherence)

def train_lda_gensim(texts, num_topics=11, passes=10):
    """
    Train LDA model using Gensim (supports coherence scoring)
    
    Parameters:
    -----------
    texts : list
        List of tokenized documents (each doc is a list of tokens)
    num_topics : int
        Number of topics
    passes : int
        Number of training passes
        
    Returns:
    --------
    tuple : (lda_model, corpus, dictionary)
    """
    # Create dictionary
    dictionary = corpora.Dictionary(texts)
    print(f"Original vocabulary: {len(dictionary)} unique tokens")
    
    # Filter extremes
    dictionary.filter_extremes(no_below=3, no_above=0.5)
    print(f"Filtered vocabulary: {len(dictionary)} unique tokens")
    
    # Create corpus (bag-of-words)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(f"Corpus size: {len(corpus)} documents")
    
    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )
    
    return lda_model, corpus, dictionary


def calculate_coherence(lda_model, texts, dictionary, coherence='c_v'):
    """
    Calculate coherence score for LDA model
    
    Parameters:
    -----------
    lda_model : gensim.models.LdaModel
        Trained LDA model
    texts : list
        List of tokenized documents
    dictionary : gensim.corpora.Dictionary
        Gensim dictionary
    coherence : str
        Coherence measure ('c_v', 'u_mass', 'c_npmi')
        
    Returns:
    --------
    float : Coherence score
    """
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence
    )
    return coherence_model.get_coherence()


def find_optimal_topics(texts, dictionary, corpus, 
                        topic_range=range(5, 25), coherence='c_v'):
    """
    Find optimal number of topics using coherence score
    
    Parameters:
    -----------
    texts : list
        List of tokenized documents
    dictionary : gensim.corpora.Dictionary
        Gensim dictionary
    corpus : list
        Gensim corpus
    topic_range : range
        Range of topic numbers to try
    coherence : str
        Coherence measure
        
    Returns:
    --------
    tuple : (best_num_topics, coherence_scores)
    """
    coherence_scores = []
    
    for num_topics in topic_range:
        print(f"Training model with {num_topics} topics...")
        
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=5,
            alpha='auto'
        )
        
        score = calculate_coherence(lda_model, texts, dictionary, coherence)
        coherence_scores.append(score)
        print(f"  Coherence ({coherence}): {score:.4f}")
    
    # Find optimal
    best_idx = np.argmax(coherence_scores)
    best_num_topics = list(topic_range)[best_idx]
    
    print(f"\nOptimal number of topics: {best_num_topics}")
    print(f"Best coherence score: {coherence_scores[best_idx]:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(list(topic_range), coherence_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel(f'Coherence Score ({coherence})')
    plt.title('Topic Coherence by Number of Topics')
    plt.axvline(x=best_num_topics, color='r', linestyle='--', 
                label=f'Optimal: {best_num_topics}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('coherence_scores.png', dpi=300)
    plt.show()
    
    return best_num_topics, coherence_scores


# 4. TOPIC-TO-FRAME MAPPING

# Frame mapping based on manual analysis of topic words
LDA_TOPIC_TO_FRAME = {
    0:  "future_focused",      # AI, humans, consciousness
    1:  "future_focused",      # jobs, automation, economic value
    2:  "technical_focused",   # social media, algorithms, platforms
    3:  "victim_focused",      # privacy, data, tracking
    4:  "technical_focused",   # language models, LLMs, training
    5:  "victim_focused",      # deepfakes, election, porn
    6:  "victim_focused",      # safety, kids, cameras, laws
    7:  "technical_focused",   # police decision-making, robots
    8:  "victim_focused",      # tiktok/youtube, age, content
    9:  "technical_focused",   # chatgpt in school, ethical/legal use
    10: "future_focused",      # crime, control, future reality
    11: "solution_focused",    # lawsuit, copyright, regulation
    12: "victim_focused",      # surveillance, fraud, identity
    13: "solution_focused",    # court, legal rights
    14: "technical_focused",   # machine learning, research
    15: "future_focused",      # China, global AI power
    16: "other",               # free speech, commentary
    17: "victim_focused",      # kids, online age, internet harms
    18: "victim_focused",      # deepfake porn
    19: "other",               # social media platforms
    20: "technical_focused",   # healthcare, medical AI
    21: "other",               # big tech power
    22: "victim_focused",      # fraud, identity, bots
    23: "technical_focused",   # self-driving cars
    24: "solution_focused",    # open source + copyright + new law
    25: "technical_focused",   # machine learning, covid
    26: "technical_focused",   # AI + art + creativity tools
    27: "future_focused",      # don't know, humans, thinking
    28: "technical_focused",   # misinformation + data training
    29: "future_focused",      # humanity, society, future
}


def assign_frame_from_topic(topic_id):
    """
    Map LDA topic to discourse frame
    
    Parameters:
    -----------
    topic_id : int
        LDA topic ID
        
    Returns:
    --------
    str : Frame label
    """
    return LDA_TOPIC_TO_FRAME.get(topic_id, "other")

# 5. MAIN EXECUTION

def main(data_path, text_col='processed_text', n_topics=30):
    """
    Main execution function
    
    Parameters:
    -----------
    data_path : str
        Path to CSV/Excel data file
    text_col : str
        Name of text column
    n_topics : int
        Number of topics for LDA
    """
    print("=" * 80)
    print("LDA TOPIC MODELING FOR AI HARM DISCOURSE")
    print("=" * 80)
    
    # Load data
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"\nLoaded {len(df)} documents")
    
    # Setup preprocessing
    stop_words, lemmatizer = setup_preprocessing()
    
    # Preprocess if needed
    if 'processed_tokens' not in df.columns:
        print("\nPreprocessing texts...")
        df['processed_tokens'] = df[text_col].apply(
            lambda x: preprocess_text(x, stop_words, lemmatizer)
        )
        df = df[df['processed_tokens'].map(len) > 0]
        print(f"Processed {len(df)} texts")
    
    # Train sklearn LDA
    print("\n" + "=" * 80)
    print("SKLEARN LDA (TF-IDF)")
    print("=" * 80)
    
    topic_assignments, topic_words, lda, vectorizer, dtm = train_lda_sklearn(
        df, text_col=text_col, n_topics=n_topics
    )
    
    df['lda_topic'] = topic_assignments
    df['frame_lda'] = df['lda_topic'].apply(assign_frame_from_topic)
    
    # Print frame distribution
    print("\n" + "=" * 80)
    print("FRAME DISTRIBUTION")
    print("=" * 80)
    print(df['frame_lda'].value_counts())
    
    # Save results
    df.to_csv('data_with_lda_topics.csv', index=False)
    print("\nResults saved to: data_with_lda_topics.csv")
    
    return df, topic_words, lda, vectorizer


if __name__ == "__main__":
    # Example usage
    # df, topics, model, vec = main('your_data.csv', n_topics=30)
    pass
