import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("VeganRecipes.csv")

# cleaning text func.
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

#cleaning the text
text_columns = ["name", "descripition", "ingredients", "steps", "Neutretion"]
for col in text_columns:
    df[col] = df[col].fillna("").apply(clean_text)

#remove unwanted numbers at the beginning
def clean_description(text):
    return re.sub(r'^\d+\s*\d*\s*', '', text)  


df["descripition"] = df["descripition"].astype(str).apply(clean_description)

# merging columns
df["combined_text"] = (
    df["ingredients"] + " " + df["steps"] + " " + df["name"] + " " + df["descripition"] + " " + df["Neutretion"]
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

# recommender function
def recommend_recipes(user_ingredients, df, vectorizer, tfidf_matrix, top_n=10):
    user_tfidf = vectorizer.transform([user_ingredients])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][-top_n:][::-1]

    
    user_ingredient_set = set(user_ingredients.split(", "))

    filtered_recipes = []
    for idx in top_indices:
        recipe_ingredients = set(df.iloc[idx]["ingredients"].split())  
        if user_ingredient_set.issubset(recipe_ingredients):  
            filtered_recipes.append(df.iloc[idx])

    filtered_df = pd.DataFrame(filtered_recipes)

    if filtered_df.empty:
        return df.iloc[top_indices][["name", "ingredients", "descripition", "steps", "Neutretion"]]

    return filtered_df[["name", "ingredients", "descripition", "steps", "Neutretion"]]

# streamlit 
st.title("Vegan Recipe Recommender")
st.write("Enter ingredients you have, and get the best vegan recipes!")

user_input = st.text_input("Enter ingredients (comma-separated):")

if user_input:
    recommendations = recommend_recipes(user_input.lower(), df, vectorizer, tfidf_matrix)

    if recommendations.empty:
        st.write("No matching recipes found. Try different ingredients.")
    else:
        st.subheader("Recommended Recipes:")
        for i, row in recommendations.iterrows():
            with st.expander(f"{row['name']}"):
                st.write(f"**Description:** {row.get('descripition', 'N/A')}")
                st.write(f"**Ingredients:** {row['ingredients']}")
                st.write(f"**Steps:** {row.get('steps', 'N/A')}")
                st.write(f"**Nutrition Info:** {row.get('Neutretion', 'N/A')}")


