# 📚 Hybrid Book Recommendation System

> An intelligent, web-based book recommendation engine that blends Collaborative Filtering and Content-Based Filtering to deliver highly accurate, personalized book suggestions.

---

## 📖 Overview
BookAI is a machine learning web application built with Streamlit. It helps users discover their next favorite book by analyzing massive datasets of user-book interactions and book metadata. Instead of relying on a single recommendation algorithm, this project implements multiple models—including K-Nearest Neighbors (KNN), Singular Value Decomposition (SVD), and TF-IDF—and fuses them into a powerful **Hybrid Engine**.

## ✨ Features
* **Multi-Algorithm Engine:** Choose between 4 different recommendation techniques directly from the UI (Hybrid, KNN, SVD, Content-Based).
* **Adjustable Output:** Dynamic slider to fetch 4, 8, or 12 recommendations at a time.
* **Modern UI/UX:** Built with custom CSS featuring a glassmorphism design, gradient themes, and responsive book grids.
* **Smart Fallbacks:** Handles missing book covers gracefully with AI-generated placeholder images.
* **Performance Optimized:** Uses cached machine learning artifacts (`.pkl` files) to ensure fast load times and instant recommendations without reading heavy `.csv` files in production.

## 🧠 Recommendation Algorithms Used

1. **K-Nearest Neighbors (KNN):** * *Type:* Memory-based Collaborative Filtering.
   * *How it works:* Calculates the cosine distance between user rating vectors to find books that have been rated similarly by the community.
2. **Singular Value Decomposition (SVD):**
   * *Type:* Model-based Collaborative Filtering.
   * *How it works:* Reduces the dimensionality of the user-item interaction matrix to identify latent underlying features and relationships between books.
3. **Content-Based Filtering (TF-IDF):**
   * *Type:* Metadata Analysis.
   * *How it works:* Uses Term Frequency-Inverse Document Frequency to vectorize text metadata (Book Title, Author, Publisher) and calculates cosine similarity to find textually similar books.
4. **Hybrid Approach (Custom):**
   * *Type:* Ensemble Method.
   * *How it works:* Applies a weighted combination of the SVD (collaborative) score and the TF-IDF (content) score (e.g., 70% SVD + 30% Content). This solves the "cold start" problem and yields the most robust recommendations.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Frontend:** Streamlit, HTML/CSS (via Markdown)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`NearestNeighbors`, `TruncatedSVD`, `TfidfVectorizer`, `cosine_similarity`)

