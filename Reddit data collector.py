#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit AI Harm Discourse Data Collector
A comprehensive tool for collecting Reddit posts and comments related to AI harm discourse
across multiple time periods and categories.
"""

import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
import warnings

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Topic modeling
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

warnings.filterwarnings('ignore', category=DeprecationWarning)

# CONFIGURATION: HARM QUERY CATEGORIES

EXTENDED_HARM_QUERIES = {


    "biometric_harms": [
        "facial recognition wrongful arrest",
        "facial recognition misidentify",
        "facial recognition police mistake",
        "clearview ai privacy",
        "clearview ai lawsuit",
        "facial recognition ban",
        "biometric surveillance",
        "emotion recognition discrimination",
        "facial recognition false match",
        "voice recognition false rejection",
        "gait recognition bias"
    ],

    "algorithmic_discrimination": [
        "ai hiring bias",
        "algorithm discrimination",
        "algorithm racist",
        "apple card gender bias",
        "compas algorithm",
        "compas algorithm bias",
        "predictive policing bias",
        "ai healthcare denial",
        "algorithm credit discrimination",
        "hiring algorithm discrimination",
        "algorithm sentencing bias",
        "ai financial fraud",
        "high frequency trading manipulation",
        "robo advisor investment loss",
        "ai mortgage denial bias",
        "amazon worker surveillance firing"

    ],

    "generative_ai_harms": [
        "deepfake non-consensual",
        "deepfake porn",
        "ai deepfake taylor swift",
        "ai voice cloning scam",
        "ai art theft",
        "midjourney lawsuit",
        "chatgpt plagiarism",
        "ai replacing artists",
        "deepfake revenge porn",
        "deepfake harassment",
        "ai election deepfake",
        "large language model halluncination harm"
    ],

    "platform_algorithm_harms": [
        "instagram algorithm mental health",
        "youtube algorithm radicalization",
        "tiktok algorithm harmful",
        "facebook algorithm misinformation",
        "social media algorithm teen",
        "content moderation error"
    ],

    "automated_system_harms": [
        "automated benefits denied",
        "proctorio false positive",
        "tenant screening algorithm",
        "automated welfare discrimination",
        "algorithm unemployment denied",
        "self-driving car accident death",
        "autonomous vehicle crash victim",
        "drone surveillance incident harm",

    ],

    "environmental_and_resource_harms": [
        "ai large model water consumption",
        "chatgpt energy use",
        "data center carbon footprint",
        "ai training heat waste",
        "cryptocurrency energy scandal"

    ],


    "legal_consequences": [
        "facial recognition lawsuit",
        "algorithm discrimination lawsuit",
        "ai lawsuit",
        "deepfake lawsuit",
        "facial recognition settlement",
        "algorithm class action",
        "sue facial recognition",
        "sued for ai bias",
        "ai patent infringement lawsuit",
        "class action against generative AI",
        "GDPR fine algorithm"

    ],

    "personal_impact": [
        "facial recognition ruined life",
        "algorithm denied job",
        "algorithm denied loan",
        "algorithm denied benefits",
        "deepfake destroyed reputation",
        "ai cost me job",
        "cant get job algorithm",
        "blacklisted by algorithm",
        "algorithm denied housing",
        "denied medical care AI",
        "deepfake identity theft"

    ],

    "systemic_impact": [
        "facial recognition mass surveillance",
        "algorithm systemic discrimination",
        "ai inequality",
        "algorithm perpetuates racism",
        "deepfake democracy threat",
        "ai misinformation social cohesion",
        "concentration of power in big tech",
        "automation job market collapse"
    ],

    "psychological_and_trust_erosion": [
        "ai system PTSD",
        "algorithm anxiety",
        "loss of trust in institutions due to AI",
        "social isolation due to platform algorithm"
    ],

    "accountability_avoidance": [
        "ai company denies responsibility",
        "algorithm failure cover up",
        "lack of recourse for algorithm victim",
        "ai audit secrecy"
    ],



    "first_person_harm": [
        "i was arrested facial recognition",
        "algorithm denied me",
        "ai discriminated against me",
        "deepfake of me",
        "facial recognition got me arrested",
        "algorithm rejected my application",
        "i lost my art to ai",
        "ai tried to scam me",
        "i was shadowbanned by algorithm"
    ],

    "victim_stories": [
        "wrongly accused facial recognition",
        "falsely arrested algorithm",
        "mistakenly identified ai",
        "victim facial recognition",
        "harmed by algorithm",
        "deepfake victim",
        "ai surveillance victim",
        "denied housing algorithm victim",
        "my identity was stolen by deepfake"
    ],

    "impact_testimony": [
        "facial recognition changed my life",
        "algorithm ruined everything",
        "cant trust ai anymore",
        "ai made me paranoid",
        "ai monitoring ruined my marriage",
        "life after deepfake"
    ],

    "empowerment_and_advocacy": [
    "how i fought the algorithm",
    "warn others about ai bias",
    "seeking help for deepfake",
    "my appeal against the algorithm"
    ],



    "moral_outrage": [
        "facial recognition outrageous",
        "algorithm unacceptable",
        "deepfake horrifying",
        "ai gone too far",
        "algorithm violation human rights",
        "facial recognition dystopia",
        "algorithm absolutely wrong",
        "scandalous ai",
        "betrayed by ai",
        "ai future uncertain",
        "privacy anxiety",
        "fear of AI replacing me"
    ],

    "concern_worry": [
        "facial recognition dangerous",
        "algorithm concerning",
        "worried about ai",
        "afraid of facial recognition",
        "algorithm scary",
        "deepfake threat",
        "ai terrifying"
    ],

    "calls_for_action": [
        "ban facial recognition",
        "regulate algorithm",
        "stop ai discrimination",
        "need facial recognition regulation",
        "algorithm accountability",
        "sue for ai discrimination",
        "ai developer code of conduct",
        "revoke facial recognition licence",
        "audit algorithm"
    ],

    "resignation_acceptance": [
        "facial recognition inevitable",
        "algorithm everywhere now",
        "cant stop ai",
        "get used to surveillance",
        "algorithm normalized",
        "accept facial recognition",
        "ai just reality now"
    ],

    "demanding_transparency": [
    "ai system transparency needed",
    "explain the algorithm decision",
    "right to explanation AI",
    "algorithm black box problem"
    ],


    "technical_failures": [
        "facial recognition error",
        "algorithm false positive",
        "ai mistake",
        "facial recognition inaccurate",
        "algorithm wrong",
        "deepfake undetectable",
        "llm hallunication mechanism",
        "model drift failure",
        "overreliance on ai prediction"
    ],

    "bias_mechanisms": [
        "facial recognition biased training data",
        "algorithm perpetuates bias",
        "ai learned discrimination",
        "biased dataset facial recognition",
        "algorithm encodes racism",
        "bias in synthetic data",
        "lack of inclusive testing",
        "historical data perpetuates inequality"
    ],

    "system_design_flaws": [
        "facial recognition no accountability",
        "algorithm black box",
        "ai unexplainable decision",
        "facial recognition no oversight",
        "lack of human in the loop",
        "ai system unpatchable vulnerability",
        "complexity makes it dangerous"
    ],

    "incentive_and_profit_mechanisms": [
      "profit over ethics AI",
      "surveillance capitalism model",
      "engagement optimization harm",
      "data hoarding danger"
      ],


    "frequency_routine": [
        "another facial recognition arrest",
        "yet another algorithm bias",
        "one more deepfake",
        "facial recognition mistake again",
        "algorithm discrimination common",
        "deepfake everywhere now",
        "another ai error",
        "another deepfake video goes viral",
        "expected an AI mistake",
        "daily algorithm failure"
    ],

    "acceptance_language": [
        "facial recognition standard now",
        "algorithm part of life",
        "ai just how it is",
        "facial recognition normal",
        "algorithm expected",
        "getting used to surveillance",
        "ai everywhere accept it",
        "cannot go back before ai",
        "it's the price of convenience",
        "ai is inevitable anyway"
    ],

    "comparative_harms": [
        "worse than facial recognition",
        "not as bad as algorithm",
        "better than human discrimination",
        "facial recognition lesser evil",
        "algorithm improves over time",
        "better than human decision",
        "ai is less corrupt than",
        "ai is just a tool"
    ],
    "personal_adaptation_and_mitigation": [
        "how to avoid facial recognition",
        "changing behavior due to AI",
        "just assume you are watched",
        "using VPN because of AI"
    ],

    "ineffective_fixes": [
        "facial recognition audit failed",
        "algorithm fairness doesnt work",
        "ai bias mitigation ineffective",
        "facial recognition oversight weak",
        "algorithm regulation toothless",
        "no legal power over algorithm",
        "band-aid solution ai",
        "ethics board ignored"
    ],

    "empty_promises": [
        "facial recognition company promises",
        "algorithm transparency claim",
        "ai ethics washing",
        "facial recognition reform insufficient",
        "algorithm accountability theater",
        "ai resposibility claim",
        "ethical ai marketing scam",
        "self regulation failure"
    ],

    "blame_and_suppression_tactics": [
        "algorithm victim blamed",
        "silence ai critics",
        "ai system gaslighting",
        "retaliation for reporting bias"
    ],


    "casual_mentions": [
        "facial recognition privacy concerns aside",
        "algorithm bias issues notwithstanding",
        "despite discrimination concerns",
        "ignoring privacy issues",
        "putting aside ethical concerns",
        "deepfake risk aside",
        "forget about the bias for a second",
        "putting aside accuracy problems"
    ],

    "justified_harms": [
        "facial recognition worth privacy trade",
        "algorithm bias acceptable cost",
        "efficiency outweighs discrimination",
        "security justifies surveillance",
        "convenience over privacy",
        "cost saving justifies algorithm",
        "better resource allocation with ai",
        "speed outweights inaccuracy"
    ],
    "inevitability_and_technological_determinism": [
        "ai is the only way forward",
        "cant avoid the algorithm",
        "must use facial recognition",
        "AI is just progress"
    ]
}


# CONFIGURATION: SUBREDDITS AND TIME PERIODS

SUBREDDITS = [
    'technology',
    'artificial',
    'MachineLearning',
    'privacy',
    'legaladvice',
    'TwoXChromosomes'
]

# Time periods
TIME_PERIODS = {
    '2020 - 2021': {
        'start': datetime(2020, 1, 1).timestamp(),
        'end': datetime(2021, 12, 31).timestamp(),
        'label': '2020 - 2021'
    },
    '2022-2023': {
        'start': datetime(2022, 1, 1).timestamp(),
        'end': datetime(2023, 12, 31).timestamp(),
        'label': '2022-2023'
    },
    '2024-2025': {
        'start': datetime(2024, 1, 1).timestamp(),
        'end': datetime(2025, 11, 3).timestamp(),  # Up to today
        'label': '2024-2025'
    }
}

# Flatten all queries into a single list with metadata
ALL_QUERIES = []
for category, queries in EXTENDED_HARM_QUERIES.items():
    for query in queries:
        ALL_QUERIES.append({
            'query': query,
            'layer': category,
            'subcategory': category
        })

# NLTK SETUP
def setup_nltk():
    """Download required NLTK data"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK resources loaded")
    except Exception as e:
        print(f"NLTK setup warning: {e}")

# DATA COLLECTION FUNCTION

def collect_reddit_data(queries_list, subreddits, time_periods, 
                       posts_per_query=20, comments_per_post=10, 
                       rate_limit_delay=2):
    """
    Collect Reddit posts and comments based on specified queries and time periods.
    
    Parameters:
    -----------
    queries_list : list
        List of query dictionaries with 'query', 'layer', and 'subcategory' keys
    subreddits : list
        List of subreddit names to search
    time_periods : dict
        Dictionary mapping period names to (start_timestamp, end_timestamp) tuples
    posts_per_query : int
        Maximum number of posts to collect per query
    comments_per_post : int
        Maximum number of top comments to collect per post
    rate_limit_delay : float
        Delay in seconds between API requests
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all collected posts and comments
    """
    import praw
    
    # Initialize Reddit API
    # NOTE: You need to set up your credentials
    reddit = praw.Reddit(
        client_id='YOUR_CLIENT_ID',
        client_secret='YOUR_CLIENT_SECRET',
        user_agent='YOUR_USER_AGENT'
    )
    
    collected_data = []
    posts_count = 0
    comments_count = 0
    
    # Statistics tracking
    stats = {period: {layer['layer']: 0 for layer in queries_list} 
             for period in time_periods.keys()}
    
    print("REDDIT DATA COLLECTION")
    print(f"Queries: {len(queries_list)}")
    print(f"Subreddits: {len(subreddits)}")
    print(f"Time periods: {len(time_periods)}")
    print(f"Posts per query: {posts_per_query}")
    print(f"Comments per post: {comments_per_post}")
    
    # Iterate through time periods
    for period_name, (period_start, period_end) in time_periods.items():
        print(f"PERIOD: {period_name}")        
        period_posts = 0
        period_comments = 0
        
        # Iterate through subreddits
        for subreddit_name in subreddits:
            print(f"\n  Subreddit: r/{subreddit_name}")
            
            # Iterate through queries
            for query_item in queries_list:
                query = query_item['query']
                layer = query_item['layer']
                subcategory = query_item['subcategory']
                
                print(f"    {layer:30s} | {query:40s} | ", end='')
                
                try:
                    # Search Reddit
                    subreddit = reddit.subreddit(subreddit_name)
                    search_results = subreddit.search(
                        query,
                        sort='relevance',
                        time_filter='all',
                        limit=posts_per_query * 3
                    )
                    
                    # Filter posts by time period
                    for post in search_results:
                        if posts_count >= posts_per_query * len(queries_list):
                            break
                            
                        if period_start <= post.created_utc <= period_end:
                            # Collect post data
                            post_data = {
                                'post_id': post.id,
                                'title': post.title,
                                'selftext': post.selftext,
                                'content_to_code': f"{post.title}. {post.selftext}",
                                'author': str(post.author),
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'created_utc': post.created_utc,
                                'created_date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
                                'subreddit': subreddit_name,
                                'url': post.url,
                                'permalink': f"https://reddit.com{post.permalink}",
                                'query': query,
                                'query_layer': layer,
                                'query_subcategory': subcategory,
                                'period': period_name,
                                'content_type': 'post',
                                'parent_id': None
                            }
                            
                            collected_data.append(post_data)
                            posts_count += 1
                            period_posts += 1
                            stats[period_name][layer] += 1
                            
                            # Collect comments from this post
                            try:
                                post.comments.replace_more(limit=0)
                                top_comments = post.comments.list()[:comments_per_post]
                                
                                for comment in top_comments:
                                    if period_start <= comment.created_utc <= period_end:
                                        comment_data = {
                                            'post_id': comment.id,
                                            'title': f"Comment on: {post.title}",
                                            'selftext': comment.body,
                                            'content_to_code': comment.body,
                                            'author': str(comment.author),
                                            'score': comment.score,
                                            'num_comments': 0,
                                            'created_utc': comment.created_utc,
                                            'created_date': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d'),
                                            'subreddit': subreddit_name,
                                            'url': f"https://reddit.com{comment.permalink}",
                                            'permalink': comment.permalink,
                                            'query': query,
                                            'query_layer': layer,
                                            'query_subcategory': subcategory,
                                            'period': period_name,
                                            'content_type': 'comment',
                                            'parent_id': post.id
                                        }
                                        
                                        collected_data.append(comment_data)
                                        comments_count += 1
                                        period_comments += 1
                                        stats[period_name][f"{layer}_comments"] += 1
                            
                            except Exception as e:
                                pass
                    
                    print(f"{posts_count} posts, {comments_count} comments")
                    time.sleep(rate_limit_delay)
                
                except Exception as e:
                    print(f"Error: {str(e)[:50]}")
                    continue
            
            print(f"     Total for {period_name}: {period_posts} posts + {period_comments} comments")
    
    # Convert to DataFrame
    df = pd.DataFrame(collected_data)
    
    # Remove duplicates
    if len(df) > 0:
        df = df.drop_duplicates(subset=['post_id'])
    
    # Print summary
    print("\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"\nTotal items collected: {len(df)}")
    print(f"Posts: {len(df[df['content_type']=='post'])}")
    print(f"Comments: {len(df[df['content_type']=='comment'])}")
    print(f"Unique items: {df['post_id'].nunique()}")
    
    return df


# DATA CLEANING AND SAVING FUNCTIONS

def clean_for_excel(text):
    """Remove illegal characters for Excel compatibility"""
    if pd.isna(text):
        return text
    
    text = str(text)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    return text

def save_collected_data(df, output_prefix='reddit_ai_harm_data'):
    """Save collected data to CSV and Excel formats with summary statistics"""
    if len(df) == 0:
        print("\nNo data to save")
        return
    
    print("\n" + "="*80)
    print("SAVING COLLECTED DATA")
    print("="*80)
    
    # Clean text columns
    print("\nCleaning text for Excel compatibility...")
    text_columns = ['title', 'selftext', 'content_to_code']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_for_excel)
    print("Text cleaned")
    
    # Save CSV
    csv_filename = f'{output_prefix}_RAW.csv'
    print(f"\nSaving CSV...")
    df.to_csv(csv_filename, index=False)
    print(f"Saved: {csv_filename}")
    
    # Save Excel
    excel_filename = f'{output_prefix}_RAW.xlsx'
    print(f"\nSaving Excel...")
    try:
        df.to_excel(excel_filename, index=False)
        print(f"Saved: {excel_filename}")
    except Exception as e:
        print(f"Excel save failed: {str(e)[:100]}")
        print("  CSV file is complete and usable!")
    
    # Print summary statistics
    print_data_summary(df)
    
    # Save crosstab
    crosstab = pd.crosstab(df['period'], df['query_layer'])
    crosstab_filename = f'{output_prefix}_layer_period_distribution.csv'
    crosstab.to_csv(crosstab_filename)
    print(f"\nSaved crosstab: {crosstab_filename}")

def print_data_summary(df):
    """Print comprehensive data summary statistics"""
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    
    print(f"\nTotal items: {len(df)}")
    print(f"  Posts: {len(df[df['content_type']=='post'])}")
    print(f"  Comments: {len(df[df['content_type']=='comment'])}")
    
    print(f"\nBy period:")
    period_counts = df['period'].value_counts().sort_index()
    for period, count in period_counts.items():
        pct = (count / len(df)) * 100
        period_df = df[df['period']==period]
        posts = len(period_df[period_df['content_type']=='post'])
        comments = len(period_df[period_df['content_type']=='comment'])
        print(f"  {period:15s}: {count:4d} ({pct:5.1f}%) - {posts}p + {comments}c")
    
    print(f"\nBy layer:")
    layer_counts = df['query_layer'].value_counts()
    for layer, count in layer_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {layer:35s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nBy subreddit:")
    sub_counts = df['subreddit'].value_counts()
    for sub, count in sub_counts.items():
        pct = (count / len(df)) * 100
        print(f"  r/{sub:20s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nBy content type:")
    type_counts = df['content_type'].value_counts()
    for content_type, count in type_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {content_type:15s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nLAYER × PERIOD DISTRIBUTION:")
    print("-"*80)
    crosstab = pd.crosstab(df['period'], df['query_layer'])
    print(crosstab)

# MAIN EXECUTION

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("REDDIT AI HARM DISCOURSE DATA COLLECTOR")
    print("="*80)
    
    # Setup
    setup_nltk()
    
    print("\nStarting data collection (posts + comments)...")
    print("This will take several hours due to rate limiting")
    print("Progress will be saved periodically\n")
    
    # Collect data
    df_collected = collect_reddit_data(
        queries_list=ALL_QUERIES,
        subreddits=SUBREDDITS,
        time_periods=TIME_PERIODS,
        posts_per_query=20,
        comments_per_post=10,
        rate_limit_delay=2
    )
    
    # Save results
    if len(df_collected) > 0:
        save_collected_data(df_collected)
    else:
        print("\nNo data collected - check API credentials and queries")
    
    print("\n" + "="*80)
    print("DATA COLLECTION SCRIPT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
