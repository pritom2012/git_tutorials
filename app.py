### 
!pip install sentence-transformers

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

def semantic_skill_search(description_df, skills_df, description_col="description_text", skill_name_col="skill_name", skill_desc_col="Skill_Descriptions", top_k=5):
    """
    Performs semantic search to find the top k skills for each description.

    Args:
        description_df (pd.DataFrame): DataFrame containing the description text.
        skills_df (pd.DataFrame): DataFrame containing skill names and descriptions.
        description_col (str): Name of the column containing description text.
        skill_name_col (str): Name of the column containing skill names.
        skill_desc_col (str): Name of the column containing skill descriptions.
        top_k (int): Number of top skills to return.

    Returns:
        pd.DataFrame: DataFrame with an added column containing top k skills and scores.
    """

    model = SentenceTransformer('all-mpnet-base-v2') # or other models

    description_embeddings = model.encode(description_df[description_col].tolist(), convert_to_tensor=True)
    skill_embeddings = model.encode(skills_df[skill_desc_col].tolist(), convert_to_tensor=True)

    top_skills_list = []

    for desc_embedding in description_embeddings:
        cosine_scores = util.cos_sim(desc_embedding, skill_embeddings)[0]
        top_results = torch.topk(cosine_scores, k=top_k)

        top_skills = []
        for score, idx in zip(top_results.values, top_results.indices):
            top_skills.append({
                "skill": skills_df[skill_name_col].iloc[idx.item()],
                "score": score.item()
            })
        top_skills_list.append(top_skills)

    description_df["top_skills"] = top_skills_list
    return description_df

# Example Usage:
# Assuming you have your dataframes loaded as description_df and skills_df

# Sample DataFrames (replace with your actual data)
data1 = {'description_text': ["Experienced software engineer with python and java skills", "Marketing professional with strong social media and communication skills", "Data Scientist with machine learning and deep learning expertise"]}
description_df = pd.DataFrame(data1)

data2 = {'skill_name': ["Python", "Java", "Social Media Marketing", "Communication", "Machine Learning", "Deep Learning", "SQL", "Project Management"],
         'Skill_Descriptions': ["Programming language used for backend development and data analysis",
                              "Object-oriented programming language",
                              "Marketing using social media platforms",
                              "Ability to convey information effectively",
                              "Algorithms that allow computers to learn from data",
                              "A subset of machine learning using neural networks",
                              "Database query language", "The discipline of organizing and completing projects"]}
skills_df = pd.DataFrame(data2)

# Run the semantic search
result_df = semantic_skill_search(description_df, skills_df)

# Print the result
print(result_df)