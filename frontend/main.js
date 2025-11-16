const API_BASE_URL = "http://localhost:8000/";
const genreSelect = document.getElementById("genreSelect");
const directorInput = document.getElementById("directorInput");
const actorInput = document.getElementById("actorInput");
const recommendBtn = document.getElementById("recommendBtn");
const movieResults = document.getElementById("movieResults");
const toast = document.getElementById("toast");
const searchForm = document.getElementById("searchForm");
const loadingBar = document.getElementById("loadingBar");

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => { toast.classList.add("hidden"); }, 3000);
}

async function fetchGenres() {
  try {
    const res = await fetch(API_BASE_URL + "api/genres");
    const data = await res.json();
    genreSelect.innerHTML = '<option value="">Select Genre</option>' + data.genres.map(
      g => `<option value="${g}">${g}</option>`
    ).join('');
  } catch {
    showToast("Unable to load genres. Try later!");
  }
}

function toggleLoading(show) {
  loadingBar.className = show ? "loading-bar active" : "loading-bar";
}

async function recommendMovies(ev) {
  if(ev) ev.preventDefault();
  movieResults.innerHTML = "";
  toggleLoading(true);
  const genre = genreSelect.value ? [genreSelect.value] : [];
  const director = directorInput.value;
  const actor = actorInput.value;
  try {
    const res = await fetch(API_BASE_URL + "api/recommend/preferences", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ genres: genre, director, actor, max_results: 12 })
    });
    if (!res.ok) throw new Error('Error fetching');
    const movies = await res.json();
    if (movies.length === 0) {
      movieResults.innerHTML = "<div class='text-center text-lg text-gray-400 mt-7'>No movies found.<br>Try another genre/director/actor.</div>";
    } else {
      movieResults.innerHTML = movies.map(m =>
        `<div class="movie-card" tabindex="0">
          <div class="movie-title">${m.title}</div>
          <div class="detail"><span class="detail-label">Genres:</span> ${m.genres}</div>
          <div class="detail"><span class="detail-label">Director:</span> ${m.director || 'Unknown'}</div>
          <div class="detail"><span class="detail-label">Cast:</span> ${m.cast || 'Unknown'}</div>
        </div>`
      ).join('');
    }
  } catch {
    showToast("Search failed. Check server/API.");
  } finally {
    toggleLoading(false);
  }
}

searchForm.addEventListener("submit", recommendMovies);
window.onload = fetchGenres;
recommendMovies();
recommendBtn.addEventListener("click", recommendMovies);
