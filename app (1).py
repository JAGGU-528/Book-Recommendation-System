"""
Hybrid Book Recommendation System
BCA Final Year Project
Developer: [Your Name]
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BookAI — Hybrid Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (glassmorphism + gradient theme)
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700;900&display=swap');

    /* ── Root Variables ── */
    :root {
        --clr-bg-start: #0f0c29;
        --clr-bg-mid:   #302b63;
        --clr-bg-end:   #24243e;
        --clr-purple:   #a855f7;
        --clr-blue:     #6366f1;
        --clr-accent:   #818cf8;
        --clr-glass:    rgba(255,255,255,0.07);
        --clr-glass-b:  rgba(255,255,255,0.15);
        --clr-text:     #f1f5f9;
        --clr-muted:    #94a3b8;
        --radius:       1rem;
        --shadow:       0 8px 32px rgba(0,0,0,0.35);
    }

    /* ── App Shell ── */
    .stApp {
        background: linear-gradient(135deg, var(--clr-bg-start) 0%, var(--clr-bg-mid) 50%, var(--clr-bg-end) 100%);
        font-family: 'Inter', sans-serif;
        color: var(--clr-text);
        min-height: 100vh;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem; max-width: 1400px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.85);
        border-right: 1px solid var(--clr-glass-b);
        backdrop-filter: blur(12px);
    }
    [data-testid="stSidebar"] * { color: var(--clr-text) !important; }

    /* ── Hero Section ── */
    .hero {
        text-align: center;
        padding: 3.5rem 1rem 2.5rem;
    }
    .hero-eyebrow {
        font-size: 0.78rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--clr-accent);
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        font-weight: 900;
        background: linear-gradient(90deg, #a78bfa, #818cf8, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 1rem;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: var(--clr-muted);
        max-width: 560px;
        margin: 0 auto 2rem;
        line-height: 1.65;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: var(--clr-glass);
        border: 1px solid var(--clr-glass-b);
        border-radius: var(--radius);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.45);
    }

    /* ── Book Card ── */
    .book-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 1.1rem;
        padding: 1.1rem;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(130,70,255,0.25);
        border-color: var(--clr-purple);
    }
    .book-cover {
        width: 100%;
        max-width: 120px;
        height: 170px;
        object-fit: cover;
        border-radius: 0.6rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    }
    .book-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .book-author {
        font-size: 0.77rem;
        color: var(--clr-muted);
        margin-bottom: 0.55rem;
    }
    .book-meta {
        font-size: 0.72rem;
        color: #64748b;
        margin-bottom: 0.4rem;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .badge-score {
        background: linear-gradient(90deg, #7c3aed, #4f46e5);
        color: #fff;
    }
    .badge-ai {
        background: rgba(99,102,241,0.25);
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,0.4);
    }

    /* ── Recommend Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 0.65rem !important;
        padding: 0.65rem 2.2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        transition: opacity 0.2s, transform 0.2s !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.45) !important;
        cursor: pointer !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-2px) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox label, .stSelectbox div { color: var(--clr-text) !important; }
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 0.65rem !important;
        color: var(--clr-text) !important;
    }

    /* ── Section Header ── */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.55rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1.25rem;
        border-left: 4px solid var(--clr-purple);
        padding-left: 0.85rem;
    }

    /* ── Stat chips in sidebar ── */
    .stat-chip {
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 0.55rem;
        padding: 0.5rem 0.85rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        color: #c7d2fe;
    }
    .stat-label { color: #818cf8; font-weight: 600; }

    /* ── Technique pill ── */
    .tech-pill {
        display: inline-block;
        background: rgba(168,85,247,0.15);
        border: 1px solid rgba(168,85,247,0.35);
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        font-size: 0.75rem;
        color: #d8b4fe;
        margin: 0.2rem;
        font-weight: 500;
    }

    /* ── Divider ── */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        margin: 2rem 0;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        color: var(--clr-muted);
        font-size: 0.8rem;
        line-height: 1.7;
    }
    .footer a { color: var(--clr-accent); text-decoration: none; }

    /* ── Spinner overlay tweak ── */
    .stSpinner > div { border-top-color: var(--clr-purple) !important; }

    /* ── Alert / Info box ── */
    .info-box {
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 0.75rem;
        padding: 1rem 1.2rem;
        color: #c7d2fe;
        font-size: 0.88rem;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all pre-trained models and data from pickle files."""
    pivot_table    = pickle.load(open("pivot_table.pkl",   "rb"))
    knn_model      = pickle.load(open("knn_model.pkl",     "rb"))
    svd_model      = pickle.load(open("svd_model.pkl",     "rb"))
    tfidf          = pickle.load(open("tfidf.pkl",         "rb"))
    content_books  = pickle.load(open("content_books.pkl", "rb"))

    # Recompute derived matrices
    book_features    = svd_model.transform(pivot_table)
    svd_similarity   = cosine_similarity(book_features)

    tfidf_matrix     = tfidf.transform(content_books["features"])
    content_similarity = cosine_similarity(tfidf_matrix)

    scaler = StandardScaler()
    pivot_normalized = scaler.fit_transform(pivot_table)
    cosine_sim       = cosine_similarity(pivot_normalized)

    return {
        "pivot_table":          pivot_table,
        "knn_model":            knn_model,
        "svd_similarity":       svd_similarity,
        "content_books":        content_books,
        "content_similarity":   content_similarity,
        "cosine_sim":           cosine_sim,
    }


# ─────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────

def recommend_hybrid(book_name: str, artifacts: dict, n: int = 8,
                     svd_weight: float = 0.7, content_weight: float = 0.3):
    """
    Hybrid recommendation combining SVD (collaborative) and
    TF-IDF content-based filtering.
    Returns list of [title, author, image_url, score].
    """
    pivot_table        = artifacts["pivot_table"]
    svd_similarity     = artifacts["svd_similarity"]
    content_books      = artifacts["content_books"]
    content_similarity = artifacts["content_similarity"]

    if book_name not in pivot_table.index:
        return None

    svd_idx = np.where(pivot_table.index == book_name)[0][0]
    cb_matches = content_books[content_books["Book-Title"] == book_name]
    if cb_matches.empty:
        return None
    content_idx = cb_matches.index[0]

    scores = []
    for i, row in content_books.iterrows():
        title = row["Book-Title"]
        if title not in pivot_table.index:
            continue
        svd_book_idx = np.where(pivot_table.index == title)[0][0]
        s_svd     = svd_similarity[svd_idx, svd_book_idx]
        s_content = content_similarity[content_idx, i]
        final     = svd_weight * s_svd + content_weight * s_content
        scores.append((title, final))

    scores.sort(key=lambda x: x[1], reverse=True)
    # exclude the query book itself
    scores = [(t, s) for t, s in scores if t != book_name][:n]

    results = []
    for title, score in scores:
        tmp = content_books[content_books["Book-Title"] == title]
        if tmp.empty:
            continue
        row = tmp.iloc[0]
        results.append([
            row["Book-Title"],
            row["Book-Author"],
            row.get("Image-URL-M", ""),
            round(float(score), 4),
            row.get("Year-Of-Publication", "N/A"),
            row.get("Publisher", "N/A"),
        ])
    return results


def recommend_knn(book_name: str, artifacts: dict, n: int = 8):
    """KNN collaborative-filtering recommendations."""
    pivot_table = artifacts["pivot_table"]
    knn_model   = artifacts["knn_model"]
    content_books = artifacts["content_books"]

    if book_name not in pivot_table.index:
        return None

    idx = np.where(pivot_table.index == book_name)[0][0]
    distances, indices = knn_model.kneighbors(
        pivot_table.iloc[idx, :].values.reshape(1, -1),
        n_neighbors=n + 1,
    )

    results = []
    for i in range(1, len(indices.flatten())):
        rec_title = pivot_table.index[indices.flatten()[i]]
        tmp = content_books[content_books["Book-Title"] == rec_title]
        if tmp.empty:
            continue
        row = tmp.iloc[0]
        similarity = round(float(1 - distances.flatten()[i]), 4)
        results.append([
            row["Book-Title"],
            row["Book-Author"],
            row.get("Image-URL-M", ""),
            similarity,
            row.get("Year-Of-Publication", "N/A"),
            row.get("Publisher", "N/A"),
        ])
    return results


def recommend_svd(book_name: str, artifacts: dict, n: int = 8):
    """SVD collaborative-filtering recommendations."""
    pivot_table    = artifacts["pivot_table"]
    svd_similarity = artifacts["svd_similarity"]
    content_books  = artifacts["content_books"]

    if book_name not in pivot_table.index:
        return None

    idx = np.where(pivot_table.index == book_name)[0][0]
    scores = sorted(enumerate(svd_similarity[idx]), key=lambda x: x[1], reverse=True)[1:n+1]

    results = []
    for i, score in scores:
        rec_title = pivot_table.index[i]
        tmp = content_books[content_books["Book-Title"] == rec_title]
        if tmp.empty:
            continue
        row = tmp.iloc[0]
        results.append([
            row["Book-Title"],
            row["Book-Author"],
            row.get("Image-URL-M", ""),
            round(float(score), 4),
            row.get("Year-Of-Publication", "N/A"),
            row.get("Publisher", "N/A"),
        ])
    return results


def recommend_content(book_name: str, artifacts: dict, n: int = 8):
    """TF-IDF content-based recommendations."""
    content_books      = artifacts["content_books"]
    content_similarity = artifacts["content_similarity"]

    matches = content_books[content_books["Book-Title"] == book_name]
    if matches.empty:
        return None

    idx    = matches.index[0]
    scores = sorted(enumerate(content_similarity[idx]), key=lambda x: x[1], reverse=True)[1:n+1]

    results = []
    for i, score in scores:
        row = content_books.iloc[i]
        results.append([
            row["Book-Title"],
            row["Book-Author"],
            row.get("Image-URL-M", ""),
            round(float(score), 4),
            row.get("Year-Of-Publication", "N/A"),
            row.get("Publisher", "N/A"),
        ])
    return results


# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────
FALLBACK_IMG = "https://via.placeholder.com/120x170/302b63/a78bfa?text=Book"

BADGE_LABELS = {
    "Hybrid":       ("🤖 AI Recommendation",  "badge-ai"),
    "KNN":          ("👥 Highly Similar",       "badge-ai"),
    "SVD":          ("⚡ Top Pick",             "badge-ai"),
    "Content-Based":("📖 Similar Content",     "badge-ai"),
}


def score_to_pct(score: float) -> str:
    """Convert 0–1 similarity score to a human-readable percentage."""
    return f"{min(int(score * 100 + 0.5), 99)}% Match"


def render_book_grid(books: list, method: str):
    """Render a responsive grid of book cards."""
    if not books:
        st.warning("No recommendations found. Try a different book title.")
        return

    badge_label, badge_cls = BADGE_LABELS.get(method, ("✨ Recommended", "badge-ai"))
    cols = st.columns(4)

    for idx, book in enumerate(books):
        title, author, img_url, score = book[0], book[1], book[2], book[3]
        year, publisher              = book[4], book[5]
        img_url = img_url if img_url else FALLBACK_IMG

        with cols[idx % 4]:
            st.markdown(
                f"""
                <div class="book-card">
                    <img class="book-cover"
                         src="{img_url}"
                         onerror="this.src='{FALLBACK_IMG}'"
                         alt="{title}">
                    <div class="book-title">{title}</div>
                    <div class="book-author">✍️ {author}</div>
                    <div class="book-meta">📅 {year} &nbsp;|&nbsp; 🏢 {publisher[:22]}{'…' if len(str(publisher)) > 22 else ''}</div>
                    <span class="badge badge-score">{score_to_pct(score)}</span>
                    <span class="badge {badge_cls}">{badge_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(artifacts):
    pivot_table   = artifacts["pivot_table"]
    content_books = artifacts["content_books"]

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem 0 1.5rem;">
                <div style="font-size:2.2rem;">📚</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.25rem;
                            font-weight:700;color:#e2e8f0;">BookAI</div>
                <div style="font-size:0.72rem;color:#818cf8;letter-spacing:0.15em;
                            text-transform:uppercase;margin-top:0.2rem;">
                    Hybrid Recommendation System
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🔬 Techniques")
        techniques = ["Cosine Similarity", "Content-Based","TF-IDF","Hybrid","SVD", "KNN"]
        for t in techniques:
            st.markdown(f'<span class="tech-pill">{t}</span>', unsafe_allow_html=True)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown("### 📊 Dataset Statistics")

        stats = {
            "📖 Books in Index":    f"{len(pivot_table.index):,}",
            "👥 Active Users":      f"{pivot_table.shape[1]:,}",
            "🧠 Content Items":     f"{len(content_books):,}",
        }
        for label, value in stats.items():
            st.markdown(
                f"""
                <div class="stat-chip">
                    <span class="stat-label">{label}</span><br>
                    <span style="font-size:1.05rem;color:#e2e8f0;">{value}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Load artifacts ──────────────────────
    with st.spinner("Initialising AI models…"):
        artifacts = load_artifacts()

    pivot_table = artifacts["pivot_table"]
    all_books   = sorted(pivot_table.index.tolist())

    # ── Sidebar ─────────────────────────────
    render_sidebar(artifacts)

    # ── Hero ────────────────────────────────
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">AI-Powered · Personalised · Accurate</div>
            <div class="hero-title">Hybrid Book<br>Recommendation System</div>
            <div class="hero-sub">
                Discover books you'll love using five ML techniques fused into
                one intelligent hybrid engine — KNN, SVD, TF-IDF,
                cosine similarity, and content-based filtering.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Search + Controls ───────────────────
    with st.container():
        col_sel, col_method, col_n = st.columns([4, 2, 1])

        with col_sel:
            selected_book = st.selectbox(
                "🔍 Search for a book",
                options=[""] + all_books,
                format_func=lambda x: "Type or scroll to select a book…" if x == "" else x,
                label_visibility="collapsed",
            )

        with col_method:
            method = st.selectbox(
                "Algorithm",
                options=["Hybrid", "KNN", "SVD", "Content-Based"],
                label_visibility="visible",
            )

        with col_n:
            n_recs = st.select_slider("Results", options=[4, 8, 12], value=8)

        recommend_btn = st.button("✨ Recommend Books", use_container_width=False)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Recommendations ─────────────────────
    if recommend_btn:
        if not selected_book:
            st.markdown(
                '<div class="info-box">⚠️  Please select a book from the dropdown first.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("🔍 Finding your next favourite book…"):
                if method == "Hybrid":
                    results = recommend_hybrid(selected_book, artifacts, n=n_recs)
                elif method == "KNN":
                    results = recommend_knn(selected_book, artifacts, n=n_recs)
                elif method == "SVD":
                    results = recommend_svd(selected_book, artifacts, n=n_recs)
                else:  # Content-Based
                    results = recommend_content(selected_book, artifacts, n=n_recs)

            if results is None:
                st.error("This book is not in our recommendation index. Please try another title.")
            else:
                st.markdown(
                    f'<div class="section-header">📚 Recommended for "{selected_book}"</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="info-box">🤖 <b>{method} algorithm</b> found '
                    f'<b>{len(results)}</b> books similar to your selection.</div>',
                    unsafe_allow_html=True,
                )
                render_book_grid(results, method)

    else:
        # ── Popular placeholder ──────────────
        st.markdown(
            '<div class="section-header">🌟 How it Works</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        cards = [
            ("🔗", "Collaborative Filtering",
             "KNN & SVD analyse user–book interaction patterns to find readers with similar tastes."),
            ("📖", "Content-Based Filtering",
             "TF-IDF vectorises book metadata — title, author, publisher — to match similar books."),
            ("⚡", "Hybrid Engine",
             "Both signals are weighted and blended into a single ranked recommendation list."),
        ]
        for col, (icon, title, desc) in zip([c1, c2, c3], cards):
            with col:
                st.markdown(
                    f"""
                    <div class="glass-card" style="text-align:center;min-height:200px;">
                        <div style="font-size:2.5rem;margin-bottom:0.7rem;">{icon}</div>
                        <div style="font-weight:700;font-size:1rem;color:#e2e8f0;
                                    margin-bottom:0.6rem;">{title}</div>
                        <div style="font-size:0.83rem;color:#94a3b8;line-height:1.6;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Footer ──────────────────────────────
    st.markdown(
        """
        <div class="footer">
            <strong style="color:#818cf8;">BookAI — Hybrid Recommendation System</strong><br>
            Built with ❤️ using <a href="https://streamlit.io">Streamlit</a> · 
            Powered by KNN · SVD · TF-IDF · Cosine Similarity<br>
            BCA Final Year Project · 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
