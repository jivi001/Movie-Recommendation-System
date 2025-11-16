from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="Tamil Movie Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

movies_df = None
tfidf_vectorizer = None
tfidf_matrix = None

class MovieResponse(BaseModel):
    movieId: int
    title: str
    genres: str
    director: Optional[str] = None
    cast: Optional[str] = None

class PreferenceRequest(BaseModel):
    genres: List[str]
    director: Optional[str] = None
    actor: Optional[str] = None
    max_results: int = 20

def load_movies_on_startup():
    global movies_df, tfidf_vectorizer, tfidf_matrix
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "Tamil-movies-cleaned.csv"
    if not csv_path.exists():
        csv_path = base_dir / "Tamil movies.csv"
    if not csv_path.exists():
        print("❌ Error: No dataset found.")
        return

    df = pd.read_csv(csv_path)
    movies_df = pd.DataFrame()
    movies_df['movieId'] = range(len(df))
    movies_df['title'] = df.get('movie_title', df.get('title', df.iloc[:, 0]))
    movies_df['genres'] = df.get('genres', pd.Series(['Unknown']*len(df)))
    movies_df['director'] = df.get('director_name', df.get('director', pd.Series(['Unknown']*len(df))))
    a1 = df.get('actor_1_name', df.get('cast', pd.Series(['']*len(df))))
    a2 = df.get('actor_2_name', pd.Series(['']*len(df)))
    a3 = df.get('actor_3_name', pd.Series(['']*len(df)))
    movies_df['cast'] = (
        a1.fillna('').astype(str) + ', ' +
        a2.fillna('').astype(str) + ', ' +
        a3.fillna('').astype(str)
    ).str.replace(r'^(,\s)+|(?:,\s)+$', '', regex=True).str.strip(', ')

    movies_df['content'] = (
        movies_df['genres'].fillna('') + ' ' +
        movies_df['director'].fillna('') + ' ' +
        movies_df['cast'].fillna('')
    )
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'[^|,\s]+')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['content'])

def get_genres_list():
    if movies_df is None:
        return []
    genres = set()
    for g in movies_df['genres'].dropna():
        genres.update([x.strip() for x in g.split(',') if x.strip()])
    return sorted(list(genres))

@app.on_event("startup")
async def startup_event():
    load_movies_on_startup()

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/index.html")

@app.get("/api/genres")
async def get_genres():
    return {"genres": get_genres_list()}

@app.post("/api/recommend/preferences", response_model=List[MovieResponse])
async def recommend_by_preferences(req: PreferenceRequest):
    filtered = movies_df
    if req.genres:
        filtered = filtered[filtered['genres'].apply(lambda x: any(g in x for g in req.genres))]
    if req.director:
        filtered = filtered[filtered['director'].str.contains(req.director, case=False, na=False)]
    if req.actor:
        filtered = filtered[filtered['cast'].str.contains(req.actor, case=False, na=False)]
    if req.genres:
        user_pref = ' '.join(req.genres)
        if req.director:
            user_pref += ' ' + req.director
        if req.actor:
            user_pref += ' ' + req.actor
        user_vector = tfidf_vectorizer.transform([user_pref])
        idxs = filtered.index
        scores = cosine_similarity(user_vector, tfidf_matrix[idxs]).flatten()
        filtered = filtered.assign(similarity=scores).sort_values('similarity', ascending=False)
    result = filtered.head(req.max_results)[['movieId', 'title', 'genres', 'director', 'cast']]
    return result.to_dict('records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)