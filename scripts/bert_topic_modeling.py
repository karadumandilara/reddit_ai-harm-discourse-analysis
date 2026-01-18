"""
BERTopic Modeling for AI Harm Discourse Analysis

This script performs BERTopic modeling on Reddit posts
about AI harm discourse, with automatic frame classification.

"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# 1. BERTOPIC MODEL CONFIGURATION

def create_optimized_bertopic(min_cluster_size=100):
    """
    Create an optimized BERTopic model with custom components
    
    Parameters:
    -----------
    min_cluster_size : int
        Minimum cluster size for HDBSCAN
        
    Returns:
    --------
    BERTopic : Configured BERTopic model
    """
    # Sentence Transformer for embeddings
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    
    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        low_memory=False
    )
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        min_samples=10
    )
    
    # CountVectorizer for topic representation
    vectorizer_model = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.5,
        max_features=500
    )
    
    # BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=15,
        verbose=True,
        calculate_probabilities=False,
        min_topic_size=50,
        nr_topics="auto"
    )
    
    return topic_model


def create_lightweight_bertopic(n_topics=30):
    """
    Create a lightweight BERTopic model for faster processing
    
    Parameters:
    -----------
    n_topics : int
        Target number of topics
        
    Returns:
    --------
    BERTopic : Configured BERTopic model
    """
    # Use smaller, faster embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=n_topics,
        language='english',
        calculate_probabilities=False,
        verbose=True
    )
    
    return topic_model


# 2. MODEL TRAINING

def train_bertopic_posts_only(df, text_col='processed_text', min_cluster_size=150):
    """
    Train BERTopic on posts only, then transform comments
    
    This approach ensures comments are assigned to post-derived topics,
    maintaining consistency in the topic structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'content_type' column distinguishing posts/comments
    text_col : str
        Name of text column
    min_cluster_size : int
        Minimum cluster size for HDBSCAN
        
    Returns:
    --------
    tuple : (df_with_topics, topic_model)
    """
    # Separate posts and comments
    df_posts = df[df["content_type"] == "post"].copy()
    df_comments = df[df["content_type"] == "comment"].copy()
    
    # Get texts
    post_docs = df_posts[text_col].astype(str).tolist()
    print(f"Number of posts: {len(post_docs)}")
    
    # Create and fit model on posts only
    topic_model = create_optimized_bertopic(min_cluster_size=min_cluster_size)
    post_topics, post_probs = topic_model.fit_transform(post_docs)
    
    # Assign topics to posts
    df_posts["bertopic"] = post_topics
    
    # Transform comments using the post-trained model
    if len(df_comments) > 0:
        comment_docs = df_comments[text_col].astype(str).tolist()
        comment_topics, _ = topic_model.transform(comment_docs)
        df_comments["bertopic"] = comment_topics
        print(f"Number of comments: {len(comment_docs)}")
    
    # Combine back
    df_combined = pd.concat([df_posts, df_comments], ignore_index=True)
    
    print(f"\nContent type distribution:")
    print(df_combined["content_type"].value_counts())
    print(f"\nUnique topics: {df_combined['bertopic'].nunique()}")
    
    return df_combined, topic_model


def train_bertopic_simple(df, text_col='processed_text', n_topics=30):
    """
    Train a simple BERTopic model on all documents
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with text data
    text_col : str
        Name of text column
    n_topics : int
        Target number of topics
        
    Returns:
    --------
    tuple : (topics, topic_model)
    """
    docs = df[text_col].astype(str).tolist()
    
    topic_model = create_lightweight_bertopic(n_topics=n_topics)
    topics, probs = topic_model.fit_transform(docs)
    
    # Display topic info
    print("\n=== BERTOPIC RESULTS ===")
    topic_info = topic_model.get_topic_info()
    print(topic_info)
    
    # Print top words for each topic
    for topic_id in sorted(set(topics)):
        if topic_id != -1:  # Skip outlier topic
            words = topic_model.get_topic(topic_id)
            if words:
                top_words = [word for word, score in words[:10]]
                print(f"\nBERTopic {topic_id}:")
                print(f"  {', '.join(top_words)}")
    
    return topics, topic_model


# 3. FRAME CLASSIFICATION


# Harm category lexicons for frame classification
HARM_CATEGORIES = {
    'victim_focused': {
        'keywords': [
            # Direct harm/victimization
            'victim', 'victims', 'harmed', 'harms', 'harmful',
            'suffered', 'suffering', 'discriminated', 'discrimination',
            'wrongful', 'unjust', 'unfair', 'oppressed', 'mistreated',
            'exploited', 'misused', 'targeted',
            
            # Physical/psychological harm
            'hurt', 'injured', 'injury', 'trauma', 'abuse', 'abused',
            'assault', 'violence', 'violent', 'harassment', 'harassed',
            'threatened', 'threats', 'fear', 'intimidation',
            
            # Law enforcement & detention
            'arrest', 'arrested', 'jailed', 'prison', 'incarcerated',
            'detained', 'detention', 'custody', 'police', 'officer',
            'officers', 'cops', 'brutality', 'shooting',
            'wrongfully accused', 'raid', 'interrogation',
            
            # Online/privacy harms
            'deepfake', 'deepfakes', 'nonconsensual', 'non-consensual',
            'revenge porn', 'leaked', 'leak', 'doxxed', 'doxxing',
            'privacy violation', 'privacy', 'stolen data',
            'scam', 'scammed', 'fraud', 'manipulated',
            
            # Economic harm
            'bankrupt', 'bankruptcy', 'financial loss', 'lost savings',
            'debt', 'foreclosure', 'evicted'
        ]
    },
    
    'technical_focused': {
        'keywords': [
            # Core technical terms
            'algorithm', 'algorithms', 'model', 'models',
            'system', 'systems', 'architecture',
            'data', 'dataset', 'datasets', 'training', 'test set',
            'accuracy', 'precision', 'recall', 'performance',
            
            # Modern AI jargon
            'optimization', 'parameters', 'weights',
            'neural', 'network', 'neural network', 'layers',
            'gpt', 'llm', 'large language model', 'transformer',
            'embeddings', 'token', 'tokens', 'tokenization',
            'prompt', 'inference', 'fine-tune', 'finetune',
            'compute', 'gpu', 'server', 'api', 'endpoint',
            
            # Software/debugging
            'code', 'coded', 'coding', 'bug', 'bugs', 'debug',
            'implementation', 'pipeline', 'deployment', 'deployed',
            
            # Evaluation
            'benchmark', 'metric', 'metrics', 'evaluation',
            'overfitting', 'underfitting', 'regularization',
            'hyperparameters'
        ]
    },
    
    'solution_focused': {
        'keywords': [
            # Policy/law/regulation
            'regulation', 'regulations', 'regulatory',
            'policy', 'policies', 'law', 'laws', 'legal',
            'ban', 'bans', 'banned', 'restrict', 'restriction',
            'compliance', 'oversight', 'governance', 'govern',
            'accountability', 'responsibility', 'liability',
            
            # Legal action
            'lawsuit', 'class action', 'settlement', 'sue', 'sues',
            'sued', 'fine', 'fines', 'penalty', 'sanctions',
            'enforcement',
            
            # Ethics/guidelines
            'ethics', 'ethical', 'guidelines', 'standards',
            'principles', 'best practices',
            'mitigate', 'mitigation', 'safeguard', 'safeguards',
            'intervention', 'risk management', 'impact assessment',
            
            # Policy actors
            'regulator', 'regulators', 'policy makers',
            'legislators', 'lawmakers', 'commission',
            'white house', 'congress', 'parliament'
        ]
    }
}


def classify_text_by_lexicon(text, categories=HARM_CATEGORIES):
    """
    Classify text into harm categories using lexicon matching
    
    Parameters:
    -----------
    text : str
        Text to classify
    categories : dict
        Dictionary of category keywords
        
    Returns:
    --------
    str : Category label
    """
    if pd.isna(text):
        return 'misc_other'
    
    text_lower = str(text).lower()
    scores = {}
    
    for category, config in categories.items():
        keywords = config['keywords']
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score
    
    # Get category with highest score
    if max(scores.values()) == 0:
        return 'misc_other'
    
    return max(scores, key=scores.get)


def map_bertopic_to_frames(topic_model, categories=HARM_CATEGORIES):
    """
    Automatically map BERTopic topics to harm frames based on keywords
    
    Parameters:
    -----------
    topic_model : BERTopic
        Trained BERTopic model
    categories : dict
        Dictionary of category keywords
        
    Returns:
    --------
    dict : Mapping from topic_id to frame
    """
    topic_to_frame = {}
    
    for topic_id in range(len(topic_model.get_topics()) - 1):
        if topic_id == -1:
            topic_to_frame[-1] = 'misc_other'
            continue
        
        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            topic_to_frame[topic_id] = 'misc_other'
            continue
        
        # Get all topic words as a string
        topic_text = ' '.join([word for word, score in topic_words[:15]])
        
        # Classify
        frame = classify_text_by_lexicon(topic_text, categories)
        topic_to_frame[topic_id] = frame
        
        print(f"Topic {topic_id}: {topic_text[:60]}... → {frame}")
    
    return topic_to_frame

# 4. VISUALIZATION

def visualize_topics(topic_model, docs, output_path='bertopic_visualization.html'):
    """
    Generate BERTopic visualizations
    
    Parameters:
    -----------
    topic_model : BERTopic
        Trained BERTopic model
    docs : list
        List of documents
    output_path : str
        Path to save visualization
    """
    # Topic visualization
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(output_path.replace('.html', '_intertopic.html'))
    
    # Barchart of top words
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_barchart.write_html(output_path.replace('.html', '_barchart.html'))
    
    # Hierarchy
    try:
        fig_hierarchy = topic_model.visualize_hierarchy()
        fig_hierarchy.write_html(output_path.replace('.html', '_hierarchy.html'))
    except:
        print("Could not generate hierarchy visualization")
    
    print(f"\nVisualizations saved to {output_path}")


# 5. MAIN EXECUTION


def main(data_path, text_col='processed_text', content_type_col='content_type'):
    """
    Main execution function
    
    Parameters:
    -----------
    data_path : str
        Path to CSV data file
    text_col : str
        Name of text column
    content_type_col : str
        Name of content type column (post/comment)
    """
    print("=" * 80)
    print("BERTOPIC MODELING FOR AI HARM DISCOURSE")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} documents")
    
    # Check if we have content_type column
    if content_type_col in df.columns:
        print("\nTraining BERTopic on posts, transforming comments...")
        df, topic_model = train_bertopic_posts_only(
            df, text_col=text_col, min_cluster_size=150
        )
    else:
        print("\nTraining BERTopic on all documents...")
        topics, topic_model = train_bertopic_simple(df, text_col=text_col)
        df['bertopic'] = topics
    
    # Map topics to frames
    print("\n" + "=" * 80)
    print("MAPPING TOPICS TO FRAMES")
    print("=" * 80)
    
    topic_to_frame = map_bertopic_to_frames(topic_model)
    df['frame_bertopic'] = df['bertopic'].map(topic_to_frame).fillna('misc_other')
    
    # Print frame distribution
    print("\n" + "=" * 80)
    print("FRAME DISTRIBUTION")
    print("=" * 80)
    print(df['frame_bertopic'].value_counts())
    
    # Save results
    df.to_csv('data_with_bertopic.csv', index=False)
    topic_model.save('bertopic_model')
    
    print("\nResults saved to: data_with_bertopic.csv")
    print("Model saved to: bertopic_model/")
    
    return df, topic_model, topic_to_frame


if __name__ == "__main__":
    # Example usage
    # df, model, mapping = main('your_data.csv')
    pass
