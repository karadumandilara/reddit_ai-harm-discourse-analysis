"""
Lexicon-Based Discourse Analysis for AI Harm Research
This script performs lexicon-based analysis to track moral vs technical
framing in AI harm discourse over time.

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# 1. SEED LEXICONS

MORAL_SEED = [
    "harm", "abuse", "violence", "danger", "risk", "hurt",
    "rights violation", "privacy violation", "privacy breach", "data leak",
    "discrimination", "bias", "racism", "sexism", "inequality",
    "vulnerable group", "victim", "exploitation", "scam", "fraud",
    "surveillance", "mass surveillance", "surveillance state"
]

TECHNICAL_SEED = [
    "accuracy", "precision", "recall", "f1 score", "auc",
    "false positive", "false negative",
    "model training", "training data", "dataset", "optimization",
    "hyperparameter", "gradient", "neural network", "transformer",
    "embedding", "attention head", "latency", "gpu", "benchmark",
    "large language model", "llm", "diffusion model"
]

VICTIM_FOCUSED_LEXICON = [
    # Direct harm/victimization
    'victim', 'victims', 'harmed', 'harms', 'harmful',
    'suffered', 'suffering', 'discriminated', 'discrimination',
    'wrongful', 'unjust', 'unfair', 'oppressed', 'mistreated',
    'exploited', 'misused', 'targeted',
    
    # Physical/psychological harm
    'hurt', 'injured', 'injury', 'trauma', 'abuse', 'abused',
    'assault', 'violence', 'violent', 'harassment', 'harassed',
    'threatened', 'threats', 'fear', 'intimidation',
    
    # Law enforcement
    'arrest', 'arrested', 'jailed', 'prison', 'incarcerated',
    'detained', 'custody', 'police', 'brutality', 'shooting',
    'wrongfully accused',
    
    # Online harms
    'deepfake', 'deepfakes', 'nonconsensual', 'revenge porn',
    'leaked', 'doxxed', 'doxxing', 'privacy violation',
    'scam', 'scammed', 'fraud', 'manipulated'
]

TECHNICAL_FOCUSED_LEXICON = [
    # Core technical
    'algorithm', 'algorithms', 'model', 'models',
    'system', 'systems', 'architecture', 'data', 'dataset',
    'training', 'accuracy', 'precision', 'recall', 'performance',
    
    # AI/ML jargon
    'optimization', 'parameters', 'weights', 'neural', 'network',
    'gpt', 'llm', 'transformer', 'embeddings', 'token',
    'prompt', 'inference', 'fine-tune', 'compute', 'gpu',
    
    # Software
    'code', 'bug', 'debug', 'implementation', 'pipeline', 'deployment',
    'benchmark', 'metric', 'evaluation', 'hyperparameters'
]

SOLUTION_FOCUSED_LEXICON = [
    'regulation', 'regulations', 'regulatory', 'policy', 'policies',
    'law', 'laws', 'legal', 'ban', 'banned', 'restrict',
    'compliance', 'oversight', 'governance', 'accountability',
    'lawsuit', 'settlement', 'fine', 'fines', 'penalty',
    'ethics', 'ethical', 'guidelines', 'standards', 'principles',
    'mitigate', 'mitigation', 'safeguard', 'intervention'
]


# 2. LEXICON EXPANSION

def expand_lexicon_from_corpus(texts, seed_terms, max_new_terms=30, 
                                ngram_range=(1, 2), min_df=10):
    """
    Expand seed lexicon using corpus co-occurrence patterns
    
    Parameters:
    -----------
    texts : list
        List of text documents
    seed_terms : list
        Initial seed terms
    max_new_terms : int
        Maximum new terms to add
    ngram_range : tuple
        N-gram range for vectorization
    min_df : int
        Minimum document frequency
        
    Returns:
    --------
    list : Expanded lexicon
    """
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        lowercase=True,
        stop_words='english',
        min_df=min_df
    )
    
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    freqs = np.asarray(X.sum(axis=0)).ravel()
    
    freq_df = pd.DataFrame({"term": vocab, "freq": freqs})
    freq_df = freq_df.sort_values("freq", ascending=False)
    
    seed_lower = [s.lower() for s in seed_terms]
    
    def is_related(term):
        for s in seed_lower:
            if s in term or term in s:
                return True
        return False
    
    candidates = freq_df[freq_df["term"].apply(is_related)]
    
    expanded = list(dict.fromkeys(seed_lower))  # Unique, order-preserving
    
    for t in candidates["term"]:
        if t not in expanded:
            expanded.append(t)
        if len(expanded) >= len(seed_lower) + max_new_terms:
            break
    
    return expanded

# 3. TEMPORAL ANALYSIS

def analyze_moral_vs_technical(df, date_col="created_utc", text_col="processed_text",
                                timestamp_unit="s", expand_lexicons=True):
    """
    Analyze moral vs technical language over time
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with text and date columns
    date_col : str
        Name of date column
    text_col : str
        Name of text column
    timestamp_unit : str
        Unit for timestamp conversion (e.g., 's' for seconds)
    expand_lexicons : bool
        Whether to expand lexicons from corpus
        
    Returns:
    --------
    tuple : (results_df, moral_terms, technical_terms, change_year)
    """
    df = df.copy()
    
    # Extract year from timestamp
    if timestamp_unit:
        df["year"] = pd.to_datetime(
            df[date_col], unit=timestamp_unit, errors="coerce"
        ).dt.year
    else:
        df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    
    # Filter years with enough data
    year_counts = df["year"].value_counts()
    valid_years = year_counts[year_counts >= 100].index.tolist()
    df = df[df["year"].isin(valid_years)]
    
    texts_all = df[text_col].fillna("").astype(str).tolist()
    
    # Expand lexicons if requested
    if expand_lexicons:
        moral_terms = expand_lexicon_from_corpus(
            texts_all, MORAL_SEED, max_new_terms=40, min_df=5
        )
        technical_terms = expand_lexicon_from_corpus(
            texts_all, TECHNICAL_SEED, max_new_terms=40, min_df=5
        )
    else:
        moral_terms = [t.lower() for t in MORAL_SEED]
        technical_terms = [t.lower() for t in TECHNICAL_SEED]
    
    # Calculate frequencies by year
    results = []
    
    for year, group in df.groupby("year", sort=True):
        texts = group[text_col].fillna("").astype(str).tolist()
        
        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            stop_words="english",
            min_df=1
        )
        X = vectorizer.fit_transform(texts)
        vocab = vectorizer.get_feature_names_out()
        freqs = np.asarray(X.sum(axis=0)).ravel()
        freq_dict = dict(zip(vocab, freqs))
        
        moral_freq = sum(freq_dict.get(t, 0) for t in moral_terms)
        technical_freq = sum(freq_dict.get(t, 0) for t in technical_terms)
        
        ratio = technical_freq / moral_freq if moral_freq > 0 else np.nan
        
        results.append({
            "year": year,
            "moral_freq": moral_freq,
            "technical_freq": technical_freq,
            "ratio_tech_over_moral": ratio
        })
    
    results_df = pd.DataFrame(results).sort_values("year").reset_index(drop=True)
    
    print("\nMORAL VS TECHNICAL LEXICON OVER TIME:")
    print(results_df)
    
    # Find change point (biggest year-over-year increase in ratio)
    ratio = results_df["ratio_tech_over_moral"]
    delta = ratio.diff()
    
    if delta.notna().any():
        idx_cp = delta.idxmax()
        change_year = results_df.loc[idx_cp, "year"]
    else:
        change_year = None
    
    return results_df, moral_terms, technical_terms, change_year


def analyze_four_frames(df, date_col="created_utc", text_col="processed_text",
                        timestamp_unit="s", expand=True):
    """
    Analyze all four discourse frames over time
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with text and date
    date_col : str
        Date column name
    text_col : str
        Text column name
    timestamp_unit : str
        Timestamp unit
    expand : bool
        Whether to expand lexicons
        
    Returns:
    --------
    tuple : (results_df, lexicons_dict, change_year)
    """
    df = df.copy()
    
    # Extract year
    if timestamp_unit:
        df["year"] = pd.to_datetime(
            df[date_col], unit=timestamp_unit, errors="coerce"
        ).dt.year
    else:
        df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    
    # Filter valid years
    year_counts = df["year"].value_counts()
    valid_years = year_counts[year_counts >= 100].index.tolist()
    df = df[df["year"].isin(valid_years)]
    
    texts_all = df[text_col].fillna("").astype(str).tolist()
    
    # Expand lexicons
    if expand:
        moral = expand_lexicon_from_corpus(texts_all, MORAL_SEED, max_new=50)
        technical = expand_lexicon_from_corpus(texts_all, TECHNICAL_SEED, max_new=50)
    else:
        moral = [t.lower() for t in MORAL_SEED]
        technical = [t.lower() for t in TECHNICAL_SEED]
    
    results = []
    
    for year, group in df.groupby("year", sort=True):
        texts = group[text_col].fillna("").astype(str).tolist()
        
        vec = CountVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english',
            min_df=1
        )
        X = vec.fit_transform(texts)
        vocab = vec.get_feature_names_out()
        freqs = np.asarray(X.sum(axis=0)).ravel()
        freq_dict = dict(zip(vocab, freqs))
        
        moral_f = sum(freq_dict.get(t, 0) for t in moral)
        tech_f = sum(freq_dict.get(t, 0) for t in technical)
        
        total_docs = len(group)
        total_tokens = X.sum()
        
        results.append({
            "year": year,
            "moral": moral_f,
            "technical": tech_f,
            "moral_norm_doc": moral_f / total_docs,
            "technical_norm_doc": tech_f / total_docs,
            "moral_norm_token": moral_f / total_tokens if total_tokens > 0 else 0,
            "technical_norm_token": tech_f / total_tokens if total_tokens > 0 else 0,
            "ratio_tech_moral": tech_f / moral_f if moral_f > 0 else np.nan
        })
    
    res = pd.DataFrame(results).sort_values("year")
    
    print("\n=== 4-FRAME TRENDS ===")
    print(res)
    
    # Find change point
    delta = res["ratio_tech_moral"].diff()
    change_year = res.loc[delta.idxmax(), "year"] if delta.notna().any() else None
    
    return res, {"moral": moral, "technical": technical}, change_year


# 4. DOCUMENT-LEVEL CLASSIFICATION

def classify_document(text, lexicons=None):
    """
    Classify a single document into discourse frame
    
    Parameters:
    -----------
    text : str
        Document text
    lexicons : dict
        Dictionary of frame lexicons
        
    Returns:
    --------
    str : Frame label
    """
    if lexicons is None:
        lexicons = {
            'victim_focused': VICTIM_FOCUSED_LEXICON,
            'technical_focused': TECHNICAL_FOCUSED_LEXICON,
            'solution_focused': SOLUTION_FOCUSED_LEXICON
        }
    
    if pd.isna(text):
        return 'other'
    
    text_lower = str(text).lower()
    scores = {}
    
    for frame, keywords in lexicons.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[frame] = score
    
    if max(scores.values()) == 0:
        return 'other'
    
    return max(scores, key=scores.get)


def classify_dataframe(df, text_col='processed_text'):
    """
    Classify all documents in a DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with text column
    text_col : str
        Name of text column
        
    Returns:
    --------
    pd.DataFrame : DataFrame with 'disc_category_lex' column
    """
    df = df.copy()
    df['disc_category_lex'] = df[text_col].apply(classify_document)
    
    print("\nFrame distribution (lexicon):")
    print(df['disc_category_lex'].value_counts())
    
    return df


# 5. VISUALIZATION


def plot_moral_vs_technical(results_df, change_year=None, save_path=None):
    """
    Plot moral vs technical language trends
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from analyze_moral_vs_technical
    change_year : int
        Year of detected change point
    save_path : str
        Path to save figure
    """
    # Matplotlib version
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw frequencies
    ax1.plot(results_df['year'], results_df['moral_freq'], 
             marker='o', label='Moral', color='#e74c3c')
    ax1.plot(results_df['year'], results_df['technical_freq'], 
             marker='s', label='Technical', color='#3498db')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Moral vs Technical Language Over Time')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    if change_year:
        ax1.axvline(x=change_year, color='black', linestyle='--', 
                    label=f'Change point: {change_year}')
    
    # Ratio
    ax2.plot(results_df['year'], results_df['ratio_tech_over_moral'], 
             marker='o', color='#9b59b6')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Technical/Moral Ratio')
    ax2.set_title('Technical to Moral Language Ratio')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=1, color='gray', linestyle=':', label='Equal ratio')
    
    if change_year:
        ax2.axvline(x=change_year, color='black', linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Plotly interactive version
    plot_df = results_df.melt(
        id_vars="year",
        value_vars=["moral_freq", "technical_freq"],
        var_name="type",
        value_name="frequency"
    )
    
    fig = px.line(
        plot_df,
        x="year",
        y="frequency",
        color="type",
        markers=True,
        title="Moral vs Technical Language Over Time"
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Frequency",
        legend_title="Type",
        template="simple_white"
    )
    
    if change_year:
        fig.add_vline(
            x=change_year,
            line_width=2,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Change point ~ {change_year}",
            annotation_position="top left"
        )
    
    fig.show()


def plot_frame_evolution(df, frame_col='disc_category_lex', period_col='period'):
    """
    Plot frame distribution over time periods
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with frame classifications
    frame_col : str
        Column with frame labels
    period_col : str
        Column with time periods
    """
    # Cross-tabulation
    frame_by_period = pd.crosstab(
        df[period_col], 
        df[frame_col], 
        normalize='index'
    ) * 100
    
    print("\nFrame distribution by period (%):")
    print(frame_by_period.round(1))
    
    # Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    
    frame_by_period.plot(kind='bar', ax=ax, colormap='viridis')
    
    plt.title('Frame Distribution Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(title='Frame', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('frame_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plotly interactive
    frame_by_year = (
        df.groupby(["year", frame_col])
          .size()
          .reset_index(name="count")
    )
    
    total_per_year = frame_by_year.groupby("year")["count"].transform("sum")
    frame_by_year["prop"] = frame_by_year["count"] / total_per_year
    
    fig = px.line(
        frame_by_year,
        x="year",
        y="prop",
        color=frame_col,
        markers=True,
        hover_data=["count"],
        title="Framing Proportions Over Time"
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Proportion",
        legend_title="Frame"
    )
    
    fig.show()


# 6. MAIN EXECUTION

def main(data_path, text_col='processed_text', date_col='created_utc'):
    """
    Main execution function
    
    Parameters:
    -----------
    data_path : str
        Path to data file
    text_col : str
        Text column name
    date_col : str
        Date column name
    """
    print("LEXICON-BASED DISCOURSE ANALYSIS")
    
    # Load data
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"\nLoaded {len(df)} documents")
    
    # Temporal analysis
    print("\n" + "=" * 80)
    print("MORAL VS TECHNICAL ANALYSIS")
    print("=" * 80)
    
    results, moral_terms, tech_terms, change_year = analyze_moral_vs_technical(
        df, date_col=date_col, text_col=text_col, 
        timestamp_unit='s', expand_lexicons=True
    )
    
    print(f"\nExpanded moral lexicon: {len(moral_terms)} terms")
    print(f"Expanded technical lexicon: {len(tech_terms)} terms")
    
    if change_year:
        print(f"\nDetected change point: {change_year}")
    
    # Document classification
    print("DOCUMENT-LEVEL CLASSIFICATION")

    df = classify_dataframe(df, text_col=text_col)
    
    # Visualization
    plot_moral_vs_technical(results, change_year, save_path='moral_vs_technical.png')
    
    # Save results
    results.to_csv('lexicon_temporal_analysis.csv', index=False)
    df.to_csv('data_with_lexicon_frames.csv', index=False)
    
    print("\nResults saved:")
    print("  - lexicon_temporal_analysis.csv")
    print("  - data_with_lexicon_frames.csv")
    
    return df, results, moral_terms, tech_terms


if __name__ == "__main__":
    # Example usage
    # df, results, moral, tech = main('your_data.csv')
    pass
