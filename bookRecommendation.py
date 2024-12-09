import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, accuracy

# Create sample user-item rating data
data = {
    'UserID': [1, 1, 2, 2, 3, 3],
    'ItemID': [1, 2, 2, 3, 3, 4],
    'Rating': [5, 4, 4, 5, 3, 4]
}
df = pd.DataFrame(data)

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['UserID', 'ItemID', 'Rating']], reader)

# Split data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.25, random_state=42)

# Train an SVD model
model = SVD()
model.fit(trainset)

# Test the model
predictions = model.test(testset)
accuracy.rmse(predictions)

# Predict rating for a specific user-item pair
user_id, item_id = 1, 3
predicted_rating = model.predict(user_id, item_id).est
print(f"Predicted rating for User {user_id} and Item {item_id}: {predicted_rating}")



#content based
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
books = pd.DataFrame({
    'BookID': [1, 2, 3, 4],
    'Title': ['The Hobbit', 'The Fellowship of the Ring', 'The Two Towers', 'The Return of the King'],
    'Genre': ['Fantasy', 'Fantasy', 'Fantasy', 'Fantasy']
})

# Use TF-IDF to vectorize the Genre column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['Genre'])

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend books based on title
def recommend_books(title, cosine_sim=cosine_sim):
    idx = books[books['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:3]  # Get top 2 similar books
    book_indices = [i[0] for i in sim_scores]
    return books['Title'].iloc[book_indices]

# Example usage
print(recommend_books('The Hobbit'))



#popularity based
# Sample book ratings dataset
ratings = pd.DataFrame({
    'BookID': [1, 2, 3, 4],
    'Title': ['The Hobbit', 'The Fellowship of the Ring', 'The Two Towers', 'The Return of the King'],
    'AverageRating': [4.8, 4.7, 4.6, 4.9],
    'RatingCount': [2000, 1500, 1800, 2500]
})

# Sort by highest average rating and popularity
popular_books = ratings.sort_values(['AverageRating', 'RatingCount'], ascending=False)

# Top 3 books
print(popular_books[['Title', 'AverageRating', 'RatingCount']].head(3))
