from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = FastAPI(title="Tamil Movie Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global variables
movies_df = None
tfidf_vectorizer = None
tfidf_matrix = None

# Models
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
    """Load Tamil movies CSV on server startup"""
    global movies_df, tfidf_vectorizer, tfidf_matrix
    
    csv_path = "Tamil-movies-cleaned.csv"
    
    # Check if cleaned version exists, otherwise use original
    if not os.path.exists(csv_path):
        csv_path = "Tamil-movies.csv"
        
        if not os.path.exists(csv_path):
            print("❌ Error: Tamil-movies.csv not found!")
            return
        
        # Clean the original CSV
        print("📊 Cleaning movie data...")
        df = pd.read_csv(csv_path)
        
        movies_df = pd.DataFrame()
        movies_df['movieId'] = range(len(df))
        movies_df['title'] = df['movie_title']
        movies_df['genres'] = df['genres'].fillna('Unknown')
        movies_df['director'] = df['director_name'].fillna('Unknown')
        movies_df['cast'] = (
            df['actor_1_name'].fillna('') + ', ' + 
            df['actor_2_name'].fillna('') + ', ' + 
            df['actor_3_name'].fillna('')
        )
        movies_df['cast'] = movies_df['cast'].str.strip(', ')
        
        # Save cleaned version
        movies_df.to_csv('Tamil-movies-cleaned.csv', index=False)
        print(f"✅ Saved cleaned data to Tamil-movies-cleaned.csv")
    else:
        print("📂 Loading cleaned movie data...")
        movies_df = pd.read_csv(csv_path)
    
    # Prepare content for TF-IDF
    movies_df['content'] = (
        movies_df['genres'].fillna('') + ' ' + 
        movies_df['director'].fillna('') + ' ' +
        movies_df['cast'].fillna('')
    )
    
    # Prepare TF-IDF vectors
    print("🔄 Preparing content vectors...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        token_pattern=r'[^|,\s]+'
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['content'])
    
    print(f"✅ Loaded {len(movies_df)} Tamil movies successfully!")
    print(f"📊 Available genres: {get_genres_list()}")

def get_genres_list():
    """Helper to get genre list"""
    if movies_df is None:
        return []
    all_genres = set()
    for genres in movies_df['genres'].dropna():
        all_genres.update(genres.split(', '))
    return sorted(list(all_genres))

# Load movies when server starts
@app.on_event("startup")
async def startup_event():
    load_movies_on_startup()

# Serve the main page
@app.get("/")
async def read_root():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/index.html")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "movies_loaded": movies_df is not None,
        "movies_count": len(movies_df) if movies_df is not None else 0
    }

@app.get("/api/genres")
async def get_genres():
    if movies_df is None:
        raise HTTPException(status_code=400, detail="No movies data loaded")
    
    return {"genres": get_genres_list()}

@app.get("/api/directors")
async def get_directors():
    if movies_df is None:
        raise HTTPException(status_code=400, detail="No movies data loaded")
    
    directors = movies_df['director'].dropna().unique().tolist()
    # Remove 'Unknown' and sort
    directors = [d for d in directors if d != 'Unknown']
    return {"directors": sorted(directors)[:100]}

@app.get("/api/actors")
async def get_actors():
    if movies_df is None:
        raise HTTPException(status_code=400, detail="No movies data loaded")
    
    all_actors = set()
    for cast in movies_df['cast'].dropna():
        all_actors.update([actor.strip() for actor in cast.split(',') if actor.strip()])
    
    return {"actors": sorted(list(all_actors))[:100]}

@app.get("/api/moods")
async def get_moods():
    moods = {
        "Happy": ["Comedy", "Romance", "Family", "Music"],
        "Excited": ["Action", "Adventure", "Thriller"],
        "Relaxed": ["Romance", "Drama", "Family"],
        "Thoughtful": ["Drama", "Mystery", "Documentary"],
        "Adventurous": ["Action", "Adventure", "Fantasy"],
        "Scared": ["Horror", "Thriller", "Mystery"]
    }
    return {"moods": list(moods.keys()), "mood_genres": moods}

@app.post("/api/recommend/preferences", response_model=List[MovieResponse])
async def recommend_by_preferences(request: PreferenceRequest):
    if movies_df is None:
        raise HTTPException(status_code=400, detail="No movies data loaded")
    
    try:
        filtered_movies = movies_df.copy()
        
        # Filter by genres
        if request.genres:
            genre_mask = filtered_movies['genres'].fillna('').apply(
                lambda x: any(genre in x for genre in request.genres)
            )
            filtered_movies = filtered_movies[genre_mask]
        
        # Filter by director
        if request.director:
            filtered_movies = filtered_movies[
                filtered_movies['director'].str.contains(request.director, case=False, na=False)
            ]
        
        # Filter by actor
        if request.actor:
            filtered_movies = filtered_movies[
                filtered_movies['cast'].str.contains(request.actor, case=False, na=False)
            ]
        
        # Calculate similarity scores
        if len(request.genres) > 0:
            user_preference = ' '.join(request.genres)
            if request.director:
                user_preference += ' ' + request.director
            if request.actor:
                user_preference += ' ' + request.actor
                
            user_vector = tfidf_vectorizer.transform([user_preference])
            
            filtered_indices = filtered_movies.index
            similarity_scores = cosine_similarity(
                user_vector, 
                tfidf_matrix[filtered_indices]
            ).flatten()
            
            filtered_movies['similarity'] = similarity_scores
            filtered_movies = filtered_movies.sort_values('similarity', ascending=False)
        
        # Return top results
        results = filtered_movies.head(request.max_results)[
            ['movieId', 'title', 'genres', 'director', 'cast']
        ]
        
        return results.to_dict('records')
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/movies/search")
async def search_movies(query: str):
    if movies_df is None:
        raise HTTPException(status_code=400, detail="No movies data loaded")
    
    mask = (
        movies_df['title'].str.contains(query, case=False, na=False) |
        movies_df['director'].str.contains(query, case=False, na=False) |
        movies_df['cast'].str.contains(query, case=False, na=False)
    )
    
    results = movies_df[mask][['title', 'genres', 'director', 'cast']].to_dict('records')
    
    return {"results": results, "count": len(results)}

@app.post("/api/recommend/similar")
async def recommend_similar(movie_title: str, n_recommendations: int = 10):
    if movies_df is None or tfidf_matrix is None:
        raise HTTPException(status_code=400, detail="No movies data loaded")
    
    try:
        movie_idx = movies_df[
            movies_df['title'].str.lower() == movie_title.lower()
        ].index
        
        if len(movie_idx) == 0:
            raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found")
        
        movie_idx = movie_idx[0]
        
        similarity_scores = cosine_similarity(
            tfidf_matrix[movie_idx:movie_idx+1], 
            tfidf_matrix
        ).flatten()
        
        similar_indices = similarity_scores.argsort()[-n_recommendations-1:-1][::-1]
        
        recommendations = movies_df.iloc[similar_indices][
            ['title', 'genres', 'director', 'cast']
        ].copy()
        recommendations['similarity_score'] = similarity_scores[similar_indices]
        
        return recommendations.to_dict('records')
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
