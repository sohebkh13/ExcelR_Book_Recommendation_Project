import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Functions for data loading
@st.cache_data
def load_data():
    users = pd.read_csv("P505/Users.csv", encoding="cp1252")
    books = pd.read_csv("P505/Books.csv", encoding="cp1252")
    ratings = pd.read_csv("P505/Ratings.csv", encoding="cp1252")
    
    # Process users data (extract country)
    for i in users:
        users['Country'] = users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\\"*$')
    users.drop('Location', axis=1, inplace=True)
    users['Country'] = users['Country'].astype('str')
    
    return users, books, ratings

# Function to display missing values
def missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_percent = round(df.isnull().mean().mul(100), 2)
    mz_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
        columns={df.index.name:'col_name', 0:'Missing Values', 1:'% of Total Values'})
    mz_table['Data_type'] = df.dtypes
    mz_table = mz_table.sort_values('% of Total Values', ascending=False)
    return mz_table.reset_index()

# Create recommendation system functions
def get_user_book_ratings(user_id, ratings_df, books_df):
    user_ratings = ratings_df[ratings_df['User-ID'] == user_id]
    user_books = pd.merge(user_ratings, books_df, on='ISBN')
    return user_books

def get_popular_books(books_df, ratings_df, n=10):
    # Get books with highest average rating and minimum number of ratings
    book_ratings = ratings_df.groupby('ISBN')['Book-Rating'].agg(['mean', 'count']).reset_index()
    # Filter books with at least 10 ratings
    popular_books = book_ratings[book_ratings['count'] >= 10].sort_values('mean', ascending=False)
    # Get book details
    popular_books = pd.merge(popular_books, books_df, on='ISBN')
    return popular_books.head(n)

def get_book_recommendations(user_id, ratings_df, books_df, n=5):
    """
    Generate personalized book recommendations using a more sophisticated collaborative filtering approach
    """
    # Get user's rated books
    user_books = get_user_book_ratings(user_id, ratings_df, books_df)
    
    if len(user_books) == 0:
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)
    
    # Calculate user similarity based on common book ratings
    user_ratings = ratings_df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating', fill_value=0)
    
    # Get the target user's ratings vector
    target_user_ratings = user_ratings.loc[user_id] if user_id in user_ratings.index else pd.Series(0, index=user_ratings.columns)
    
    # Find similar users
    user_similarities = []
    for other_user in user_ratings.index:
        if other_user != user_id:
            # Calculate cosine similarity between rating vectors
            other_user_ratings = user_ratings.loc[other_user]
            
            # Get common books (non-zero ratings)
            common_books = target_user_ratings.multiply(other_user_ratings) != 0
            common_count = common_books.sum()
            
            # Only consider users with at least one common book
            if common_count > 0:
                # Calculate similarity score
                similarity = np.dot(target_user_ratings, other_user_ratings) / (
                    np.sqrt(np.dot(target_user_ratings, target_user_ratings)) * 
                    np.sqrt(np.dot(other_user_ratings, other_user_ratings))
                )
                if not np.isnan(similarity):
                    user_similarities.append((other_user, similarity, common_count))
    
    # Sort similar users by similarity score
    user_similarities.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Take top 20 similar users
    similar_users = [user for user, _, _ in user_similarities[:20]]
    
    # Get books rated by similar users but not by the target user
    user_rated_isbns = set(user_books['ISBN'])
    recommendations = {}
    
    # Find books rated highly by similar users
    for idx, similar_user in enumerate(similar_users):
        # Get books rated by this similar user
        similar_user_books = ratings_df[ratings_df['User-ID'] == similar_user]
        
        # Weight factor decreases as we go down the similarity list
        weight = 1.0 / (idx + 1.5)  # Diminishing weight for less similar users
        
        # Add weighted rating to recommendations
        for _, row in similar_user_books.iterrows():
            isbn = row['ISBN']
            rating = row['Book-Rating']
            
            # Skip books the target user has already rated
            if isbn in user_rated_isbns:
                continue
                
            # Skip zero ratings
            if rating == 0:
                continue
                
            # Add weighted rating to recommendations
            if isbn not in recommendations:
                recommendations[isbn] = {'weighted_sum': 0, 'count': 0}
            
            recommendations[isbn]['weighted_sum'] += rating * weight
            recommendations[isbn]['count'] += 1
    
    # Calculate final scores
    for isbn in recommendations:
        recommendations[isbn]['score'] = recommendations[isbn]['weighted_sum'] / recommendations[isbn]['count']
    
    # Sort by score
    sorted_recommendations = sorted(recommendations.items(), 
                                   key=lambda x: (x[1]['score'], x[1]['count']), 
                                   reverse=True)
    
    # Get top N recommendations
    top_isbns = [isbn for isbn, _ in sorted_recommendations[:n*2]]  # Get more than needed to ensure we have enough after merge
    
    # Get book details
    if top_isbns:
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()
        
        # Add the score information
        for isbn in top_isbns:
            if isbn in recommendations:
                mask = recommended_books['ISBN'] == isbn
                recommended_books.loc[mask, 'mean'] = recommendations[isbn]['score']
                recommended_books.loc[mask, 'count'] = recommendations[isbn]['count']
        
        # Sort and get top N
        recommended_books = recommended_books.sort_values('mean', ascending=False).head(n)
        return recommended_books
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)

# Add this simplified recommendation function above the main() function

def get_book_recommendations_simple(user_id, ratings_df, books_df, n=5, progress_bar=None):
    """
    Generate personalized book recommendations using a simplified approach
    that's more efficient for a large dataset
    """
    # Get user's rated books
    user_books = get_user_book_ratings(user_id, ratings_df, books_df)
    user_rated_isbns = set(user_books['ISBN'])
    
    if len(user_books) == 0:
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)
    
    if progress_bar:
        progress_bar.progress(10)
    
    # Find users who rated at least one book that the target user has also rated
    common_users = ratings_df[ratings_df['ISBN'].isin(user_rated_isbns)]['User-ID'].unique()
    common_users = [u for u in common_users if u != user_id]
    
    if progress_bar:
        progress_bar.progress(30)
    
    if len(common_users) == 0:
        return get_popular_books(books_df, ratings_df, n)
    
    # Limit to a reasonable number of similar users for performance
    if len(common_users) > 100:
        common_users = common_users[:100]
    
    # Get all ratings from these users
    similar_users_ratings = ratings_df[ratings_df['User-ID'].isin(common_users)]
    
    if progress_bar:
        progress_bar.progress(50)
    
    # Filter out books the user has already rated
    candidate_books = similar_users_ratings[~similar_users_ratings['ISBN'].isin(user_rated_isbns)]
    
    # Group by ISBN and calculate average rating and count
    book_stats = candidate_books.groupby('ISBN').agg(
        score=('Book-Rating', 'mean'),
        supporting_users=('User-ID', 'nunique')
    ).reset_index()
    
    if progress_bar:
        progress_bar.progress(70)
    
    # Sort by score and number of supporting users
    book_stats = book_stats.sort_values(['score', 'supporting_users'], ascending=False)
    
    # Get top N books
    top_isbns = book_stats.head(n*2)['ISBN'].tolist()  # Get more than needed to ensure we have enough after merge
    
    if progress_bar:
        progress_bar.progress(90)
    
    # Get book details
    if top_isbns:
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()
        # Merge with scores
        recommended_books = pd.merge(recommended_books, book_stats, on='ISBN')
        # Sort and get top N
        recommended_books = recommended_books.sort_values(['score', 'supporting_users'], ascending=False).head(n)
        return recommended_books
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)

# Main app
def main():
    st.title("📚 Book Recommendation System")
    
    # Load data
    with st.spinner("Loading data... Please wait."):
        users, books, ratings = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Home", "Data Exploration", "User Analysis", "Book Recommendations", "About"])
    
    if page == "Home":
        st.header("Welcome to the Book Recommendation System!")
        st.write("""
        This application helps you discover new books to read based on user ratings and preferences.
        Use the sidebar to navigate through different sections of the app.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.write(f"Number of Users: {users.shape[0]}")
            st.write(f"Number of Books: {books.shape[0]}")
            st.write(f"Number of Ratings: {ratings.shape[0]}")
        
        with col2:
            st.subheader("Sample Books")
            st.dataframe(books.sample(5)[['Book-Title', 'Book-Author', 'Year-Of-Publication']])
        
        st.subheader("Popular Books")
        popular_books = get_popular_books(books, ratings)
        st.dataframe(popular_books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'mean', 'count']])
    
    elif page == "Data Exploration":
        st.header("Data Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Books", "Users", "Ratings"])
        
        with tab1:
            st.subheader("Books Dataset")
            st.dataframe(books.head())
            
            st.subheader("Missing Values in Books Dataset")
            st.dataframe(missing_values(books))
            
            st.subheader("Books Published by Year")
            # Fixed this section to properly handle value_counts() result
            years_df = pd.DataFrame(books['Year-Of-Publication'].value_counts()).reset_index()
            # Explicitly rename columns to match what we expect
            years_df.columns = ['Year', 'Count']
            # Convert Year to numeric for proper filtering and sorting
            years_df['Year'] = pd.to_numeric(years_df['Year'], errors='coerce')
            # Sort by year and filter valid years
            years_df = years_df.sort_values('Year').reset_index(drop=True)
            valid_years = years_df[(years_df['Year'] >= 1900) & (years_df['Year'] <= 2010)]
            
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            sns.lineplot(data=valid_years, x='Year', y='Count', ax=ax)
            plt.title('Number of Books Published by Year')
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Users Dataset")
            st.dataframe(users.head())
            
            st.subheader("Missing Values in Users Dataset")
            st.dataframe(missing_values(users))
            
            st.subheader("Age Distribution")
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100], ax=ax)
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Count')
            st.pyplot(fig)
            
            st.subheader("Top 10 Countries")
            country_counts = users['Country'].value_counts().head(10)
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            sns.barplot(x=country_counts.index, y=country_counts.values, ax=ax)
            plt.title('Top 10 Countries')
            plt.xlabel('Country')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Ratings Dataset")
            st.dataframe(ratings.head())
            
            st.subheader("Rating Distribution")
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            sns.countplot(x='Book-Rating', data=ratings, ax=ax)
            plt.title('Rating Distribution')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            st.pyplot(fig)
    
    elif page == "User Analysis":
        st.header("User Analysis")
        
        # User selection
        user_id = st.selectbox("Select User ID", sorted(users['User-ID'].unique()))
        
        if user_id:
            # Show user info
            user_info = users[users['User-ID'] == user_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("User Information")
                st.write(f"**User ID:** {user_id}")
                st.write(f"**Age:** {user_info['Age'].values[0] if not pd.isna(user_info['Age'].values[0]) else 'Not specified'}")
                st.write(f"**Country:** {user_info['Country'].values[0] if user_info['Country'].values[0] != 'nan' else 'Not specified'}")
            
            # Show user's book ratings
            user_books = get_user_book_ratings(user_id, ratings, books)
            
            with col2:
                st.subheader("Rating Statistics")
                st.write(f"**Number of Books Rated:** {len(user_books)}")
                if len(user_books) > 0:
                    avg_rating = user_books['Book-Rating'].mean()
                    st.write(f"**Average Rating:** {avg_rating:.2f}")
            
            if len(user_books) > 0:
                st.subheader("Books Rated by User")
                user_books_display = user_books.sort_values('Book-Rating', ascending=False)[
                    ['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Book-Rating']
                ]
                st.dataframe(user_books_display)
                
                st.subheader("Rating Distribution")
                fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
                sns.countplot(x='Book-Rating', data=user_books, ax=ax)
                plt.title(f'Rating Distribution for User {user_id}')
                plt.xlabel('Rating')
                plt.ylabel('Count')
                st.pyplot(fig)
            else:
                st.warning("This user has not rated any books.")
    
    elif page == "Book Recommendations":
        st.header("Book Recommendations")
        
        tab1, tab2 = st.tabs(["Get Personalized Recommendations", "View User's Top Rated Books"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Get Recommendations")
                # Create a selectbox with user IDs instead of manual input
                user_ids = sorted(users['User-ID'].unique())
                user_id = st.selectbox("Select User ID", user_ids)
                num_recommendations = st.slider("Number of Recommendations", 1, 20, 5)
                recommend_button = st.button("Get Recommendations")
            
            if recommend_button:
                with col2:
                    st.subheader(f"Top {num_recommendations} Book Recommendations for User {user_id}")
                    
                    with st.spinner("Finding the best books for you..."):
                        # Add a progress bar for better UX during calculations
                        progress_bar = st.progress(0)
                        
                        try:
                            # Try to get personalized recommendations
                            recommended_books = get_book_recommendations_simple(user_id, ratings, books, num_recommendations, progress_bar)
                            progress_bar.progress(100)
                            
                            if len(recommended_books) > 0:
                                # Display each recommended book with image
                                for i, (idx, book) in enumerate(recommended_books.iterrows()):
                                    col_img, col_info = st.columns([1, 3])
                                    
                                    with col_img:
                                        # Handle missing or invalid image URLs
                                        try:
                                            if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                                st.image(book['Image-URL-M'], use_column_width=True)
                                            else:
                                                st.write("📚 No image available")
                                        except Exception:
                                            st.write("📚 Image unavailable")
                                    
                                    with col_info:
                                        st.subheader(f"{i+1}. {book['Book-Title']}")
                                        st.write(f"**Author:** {book['Book-Author']}")
                                        st.write(f"**Published:** {book['Year-Of-Publication']}")
                                        st.write(f"**Publisher:** {book['Publisher']}")
                                        
                                        # Handle different score column names
                                        if 'score' in book:
                                            st.write(f"**Recommendation Score:** {book['score']:.2f}")
                                            st.write(f"**Based on:** {int(book['supporting_users'])} similar users")
                                        elif 'mean' in book:
                                            st.write(f"**Average Rating:** {book['mean']:.2f}")
                                            st.write(f"**Based on:** {int(book['count'])} ratings")
                            else:
                                st.warning("Not enough data to make recommendations for this user. Showing popular books instead.")
                                popular_books = get_popular_books(books, ratings, num_recommendations)
                                st.dataframe(popular_books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'mean', 'count']])
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            st.info("Showing popular books instead.")
                            popular_books = get_popular_books(books, ratings, num_recommendations)
                            
                            # Display popular books in a more visual way
                            for i, (idx, book) in enumerate(popular_books.iterrows()):
                                col_img, col_info = st.columns([1, 3])
                                
                                with col_img:
                                    try:
                                        if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                            st.image(book['Image-URL-M'], use_column_width=True)
                                        else:
                                            st.write("📚 No image available")
                                    except Exception:
                                        st.write("📚 Image unavailable")
                                
                                with col_info:
                                    st.subheader(f"{i+1}. {book['Book-Title']}")
                                    st.write(f"**Author:** {book['Book-Author']}")
                                    st.write(f"**Published:** {book['Year-Of-Publication']}")
                                    st.write(f"**Publisher:** {book['Publisher']}")
                                    st.write(f"**Average Rating:** {book['mean']:.2f}")
                                    st.write(f"**Based on:** {int(book['count'])} ratings")
        
        with tab2:
            st.subheader("User's Top Rated Books")
            # Create a selectbox with user IDs
            user_ids = sorted(users['User-ID'].unique())
            selected_user_id = st.selectbox("Select User ID", user_ids, key="top_rated_user_select")
            max_books = st.slider("Maximum Number of Books to Show", 1, 20, 10)
            
            if st.button("Show Top Rated Books"):
                # Get user's book ratings
                user_books = get_user_book_ratings(selected_user_id, ratings, books)
                
                if len(user_books) > 0:
                    st.write(f"User {selected_user_id} has rated {len(user_books)} books.")
                    
                    # Sort by rating and display top N
                    top_rated = user_books.sort_values('Book-Rating', ascending=False).head(max_books)
                    
                    # Display each book
                    for i, (idx, book) in enumerate(top_rated.iterrows()):
                        col_img, col_info = st.columns([1, 3])
                        
                        with col_img:
                            st.image(book['Image-URL-M'], use_column_width=True)
                        
                        with col_info:
                            st.subheader(f"{i+1}. {book['Book-Title']}")
                            st.write(f"**Author:** {book['Book-Author']}")
                            st.write(f"**Published:** {book['Year-Of-Publication']}")
                            st.write(f"**User's Rating:** {int(book['Book-Rating'])} / 10")
                else:
                    st.warning(f"User {selected_user_id} has not rated any books.")
    
    elif page == "About":
        st.header("About This Project")
        st.write("""
        ## Book Recommendation System
        
        This book recommendation system was developed as part of ExcelR's data science project. The system analyzes user ratings 
        to recommend books that users might enjoy based on their previous preferences and similar users' ratings.
        
        ### Dataset
        
        The dataset includes:
        - Books information (title, author, year, publisher)
        - User demographics (age, country)
        - User ratings for books
        
        ### Methods
        
        The recommendation system uses collaborative filtering techniques to find books that might interest users based on 
        the ratings of similar users.
        
        ### Features
        
        - Data exploration and visualization
        - User analysis
        - Book recommendations based on user preferences
        - Popular book recommendations
        
        ### Developer
        
        - **Developer:** Soheb
        - **GitHub:** [sohebkh13/ExcelR_Book_Recommendation_Project](https://github.com/sohebkh13/ExcelR_Book_Recommendation_Project)
        """)

if __name__ == "__main__":
    main()