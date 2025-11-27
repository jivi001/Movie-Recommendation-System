const API_BASE_URL = "/"; // Relative path to avoid CORS/port issues
const genreSelect = document.getElementById("genreSelect");
const directorInput = document.getElementById("directorInput");
const actorInput = document.getElementById("actorInput");
const recommendBtn = document.getElementById("recommendBtn");
const resultsSection = document.getElementById("resultsSection");
const toast = document.getElementById("toast");
const searchForm = document.getElementById("searchForm");
const loadingIndicator = document.getElementById("loadingIndicator");

// Toast Notification
function showToast(msg, type = 'error') {
    toast.textContent = msg;
    toast.style.borderLeftColor = type === 'error' ? '#f43f5e' : '#10b981';
    toast.classList.remove("hidden");
    setTimeout(() => {
        toast.classList.add("hidden");
    }, 3000);
}

// Fetch Genres on Load
async function fetchGenres() {
    try {
        const res = await fetch(API_BASE_URL + "api/genres");
        if (!res.ok) throw new Error("Failed to fetch genres");
        const data = await res.json();
        
        // Clear existing options except the first one
        genreSelect.innerHTML = '<option value="" disabled selected>Select a Genre</option>';
        
        data.genres.forEach(g => {
            const option = document.createElement("option");
            option.value = g;
            option.textContent = g;
            genreSelect.appendChild(option);
        });
    } catch (error) {
        console.error(error);
        showToast("Unable to load genres. Please refresh the page.");
    }
}

// Toggle Loading State
function toggleLoading(show) {
    if (show) {
        loadingIndicator.classList.remove("hidden");
        resultsSection.classList.add("hidden");
    } else {
        loadingIndicator.classList.add("hidden");
        resultsSection.classList.remove("hidden");
    }
}

// Render Movie Cards
function renderMovies(movies) {
    resultsSection.innerHTML = "";
    
    if (movies.length === 0) {
        resultsSection.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">🤔</span>
                <h3>No movies found</h3>
                <p>Try adjusting your filters to find more results.</p>
            </div>
        `;
        return;
    }

    movies.forEach(m => {
        const card = document.createElement("div");
        card.className = "movie-card";
        card.innerHTML = `
            <h3 class="movie-title">${m.title}</h3>
            <div class="movie-meta">
                <div class="meta-item">
                    <span class="meta-label">Genre:</span>
                    <span>${m.genres}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Director:</span>
                    <span>${m.director || 'Unknown'}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Cast:</span>
                    <span>${m.cast || 'Unknown'}</span>
                </div>
            </div>
        `;
        resultsSection.appendChild(card);
    });
}

// Handle Recommendation
async function recommendMovies(ev) {
    if (ev) ev.preventDefault();
    
    toggleLoading(true);
    
    const genre = genreSelect.value;
    const director = directorInput.value.trim();
    const actor = actorInput.value.trim();
    
    // Basic validation: at least one filter should be active
    if (!genre && !director && !actor) {
        toggleLoading(false);
        // If no filters, maybe show top rated or random? For now, just show a message or fetch all/default
        // Let's fetch default recommendations if nothing selected, or just warn user.
        // The backend handles empty filters by returning top matches or similar.
    }

    const payload = {
        genres: genre ? [genre] : [],
        director: director || null,
        actor: actor || null,
        max_results: 12
    };

    try {
        const res = await fetch(API_BASE_URL + "api/recommend/preferences", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error('Error fetching recommendations');
        
        const movies = await res.json();
        renderMovies(movies);
        
    } catch (error) {
        console.error(error);
        showToast("Search failed. Please try again.");
    } finally {
        toggleLoading(false);
    }
}

// Event Listeners
searchForm.addEventListener("submit", recommendMovies);

// Initial Load
window.addEventListener('DOMContentLoaded', () => {
    fetchGenres();
    // Optionally load some initial recommendations
    recommendMovies();
});
