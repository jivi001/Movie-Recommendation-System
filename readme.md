Tamil Movie Recommendation System
Discover and explore the best Tamil movies, using filters for genre, director, and actor, with smart recommendations powered by FastAPI, Python, and a modern frontend UI.

Features
🎬 Smart Recommendations: Get suggestions based on your preferred genre, director, and actor.

⚡️ FastAPI Backend: Rapid, scalable API for movie search and matching.

🍿 Premium Frontend Experience: Polished, mobile-friendly UI with attractive cards, responsive layout, and smooth animations.

📊 Advanced Search: Uses NLP (TF-IDF vectorizer) and similarity scoring for best match ranking.

📝 Easy Setup: Designed for students, developers, and movie fans.

Screenshots
(Add screenshots here of completed UI and recommendation results)

Technologies Used
Backend: Python, FastAPI, Pandas, scikit-learn, Uvicorn

Frontend: HTML5, CSS3, Tailwind CSS, Vanilla JavaScript

Data Storage: CSV files (Tamil-movies.csv, Tamil-movies-cleaned.csv)

Installation & Setup
1. Clone the Repository
bash
git clone https://github.com/yourusername/tamil-movie-rec.git
cd tamil-movie-rec
2. Setup Environment
Install Python (recommended 3.9+)

Create a virtual environment:

bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
3. Install Requirements
bash
pip install -r requirements.txt
4. Prepare Dataset
Place Tamil-movies.csv in the project root.

On first run, a cleaned version will be generated automatically as Tamil-movies-cleaned.csv.

5. Run Backend Server
bash
uvicorn movie_api:app --reload
Access the app at http://localhost:8000/

Folder Structure
text
Tamil-Movie-Rec/
│
├── movie_api.py                # FastAPI backend
├── requirements.txt            # Python dependencies
├── Tamil-movies.csv            # Raw movies dataset
├── Tamil-movies-cleaned.csv    # Auto-generated cleaned dataset
├── frontend/                   # Frontend static files
│   ├── index.html
│   ├── style.css
│   ├── main.js
│   ├── manifest.json           # For PWA support (optional)
│   └── sw.js                   # Service Worker (optional)
└── ...
Usage
Browse to http://localhost:8000/ after starting the backend.

Select Genre: Choose from 30+ genres found in Tamil cinema.

Advanced Filters: Optionally enter director/actor names for smarter recommendations.

Enjoy Results: Browse how your preferences yield relevant movie cards.

Customization
Update UI design in style.css for colors, icons, and layout.

Add more filters (year, ratings, etc.) in both API and frontend.

Replace or update dataset as needed.

API Endpoints
Endpoint	Method	Description
/api/genres	GET	List of all available genres
/api/recommend/preferences	POST	Get recommendations by user prefs
/api/directors	GET	List of directors
/api/actors	GET	List of actors
/api/movies/search?query=xxx	GET	Search movies by string
/	GET	Serve main frontend
Contributing
Pull requests, feedback, and new dataset ideas are welcome!

See the code comments and backend endpoints for extension points.

License
MIT License. See LICENSE file.

Credits
Movie dataset: Various public sources (IMDB, Wikipedia, Enthusiast Collections)

Developer: [jivitesh / https://github.com/jivi001]