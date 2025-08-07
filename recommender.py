import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os
import re

class IndianRecipeRecommender:
    def __init__(self, recipes_path):
        # Check if the file exists
        if not os.path.exists(recipes_path):
            raise FileNotFoundError(f"File '{recipes_path}' not found")

        # Load the dataset
        try:
            self.recipes_df = pd.read_csv(recipes_path)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

        # Check required columns
        required_columns = ['TranslatedRecipeName', 'TranslatedIngredients', 'TranslatedInstructions']
        for col in required_columns:
            if col not in self.recipes_df.columns:
                raise ValueError(
                    f"Required column '{col}' not found in the dataset. "
                    f"Available columns: {', '.join(self.recipes_df.columns)}"
                )

        # Use clean canonical ingredient list column (adjust column name if needed)
        clean_col = None
        for possible in ['NER', 'CleanIngredients', 'Ingredients_Clean', 'Ingredients_NER']:
            if possible in self.recipes_df.columns:
                clean_col = possible
                break
        if not clean_col:
            # Fallback: use last column if it looks like a comma-separated ingredient list
            last_col = self.recipes_df.columns[-3]  # -3 to skip image and rating columns
            if self.recipes_df[last_col].astype(str).str.contains(',').all():
                clean_col = last_col
            else:
                raise ValueError("Could not find a clean ingredient list column in your CSV.")

        self.clean_ingredient_col = clean_col

        self._preprocess_data()
        self.all_ingredients = self._extract_all_ingredients()
        self._prepare_model()

    def _preprocess_data(self):
        self.recipes_df['RecipeName'] = self.recipes_df['TranslatedRecipeName'].astype(str).str.strip()
        # Use clean list for all matching
        self.recipes_df['CleanIngredientsList'] = self.recipes_df[self.clean_ingredient_col].apply(
            lambda x: [i.strip().lower() for i in str(x).split(',')] if pd.notna(x) else []
        )
        # For display, parse original ingredient text
        self.recipes_df['IngredientsList'] = self.recipes_df['TranslatedIngredients'].apply(self._parse_ingredients)
        self.recipes_df['IngredientsText'] = self.recipes_df['CleanIngredientsList'].apply(lambda x: ' '.join(x))
        # Parse instructions into steps
        self.recipes_df['InstructionsList'] = self.recipes_df['TranslatedInstructions'].apply(self._parse_instructions)

    def _parse_ingredients(self, ingredients_text):
        if pd.isna(ingredients_text):
            return []
        ingredients = [ing.strip() for ing in str(ingredients_text).split(',')]
        return ingredients

    def _parse_instructions(self, instructions_text):
        if pd.isna(instructions_text):
            return []
        steps = [step.strip() for step in re.split(r'\.\s+(?=[A-Z])|\n', str(instructions_text)) if step.strip()]
        return steps

    def _extract_all_ingredients(self):
        all_ingredients = set()
        for ingredients in self.recipes_df['CleanIngredientsList']:
            all_ingredients.update(ingredients)
        return list(all_ingredients)

    def _prepare_model(self):
        self.vectorizer = CountVectorizer(binary=True)
        X = self.vectorizer.fit_transform(self.recipes_df['IngredientsText'])
        y = np.arange(len(self.recipes_df))
        self.model = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000)
        self.model.fit(X, y)

    def recommend_recipes(self, input_ingredients, top_n=5):
        # Clean and preprocess input ingredients
        cleaned_ingredients = [i.strip().lower() for i in input_ingredients if i.strip()]
        if not cleaned_ingredients:
            return []

        input_text = ' '.join(cleaned_ingredients)
        input_vector = self.vectorizer.transform([input_text])
        recipe_probs = self.model.predict_proba(input_vector)[0]
        top_indices = recipe_probs.argsort()[-top_n:][::-1]

        recommendations = []
        for idx in top_indices:
            recipe = self.recipes_df.iloc[idx]
            score = recipe_probs[idx]
            if score > 0:
                recommendations.append({
                    'title': recipe['RecipeName'],
                    'confidence': round(score * 100, 2),
                    'ingredients': recipe['TranslatedIngredients'],
                    'missing_ingredients': self.get_missing_ingredients(cleaned_ingredients, recipe['CleanIngredientsList']),
                    'instructions': recipe['InstructionsList']
                })
        return recommendations

    def get_missing_ingredients(self, input_ingredients, recipe_ingredients):
        input_set = set([i.strip().lower() for i in input_ingredients])
        missing = [ing for ing in recipe_ingredients if ing not in input_set]
        return missing
