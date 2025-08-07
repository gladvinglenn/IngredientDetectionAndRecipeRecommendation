import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import argparse
import sys
import os
import re

class IndianRecipeRecommender:
    def __init__(self, recipes_path):
        # Check if the file exists
        if not os.path.exists(recipes_path):
            print(f"Error: File '{recipes_path}' not found")
            sys.exit(1)

        # Load the dataset
        try:
            self.recipes_df = pd.read_csv(recipes_path)
            print(f"Loaded {len(self.recipes_df)} Indian recipes.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

        # Check required columns
        required_columns = ['TranslatedRecipeName', 'TranslatedIngredients', 'TranslatedInstructions']
        for col in required_columns:
            if col not in self.recipes_df.columns:
                print(f"Error: Required column '{col}' not found in the dataset")
                print(f"Available columns: {', '.join(self.recipes_df.columns)}")
                sys.exit(1)

        # Use clean canonical ingredient list column (adjust column name if needed)
        # Try to find the clean ingredient list column (NER or similar)
        clean_col = None
        for possible in ['NER', 'CleanIngredients', 'Ingredients_Clean', 'Ingredients_NER']:
            if possible in self.recipes_df.columns:
                clean_col = possible
                break
        if not clean_col:
            # Fallback: use last column if it looks like a comma-separated ingredient list
            last_col = self.recipes_df.columns[-3]  # -3 to skip image and rating columns
            if self.recipes_df[last_col].str.contains(',').all():
                clean_col = last_col
            else:
                print("Error: Could not find a clean ingredient list column in your CSV.")
                sys.exit(1)

        self.clean_ingredient_col = clean_col

        self._preprocess_data()
        self.all_ingredients = self._extract_all_ingredients()
        self._prepare_model()
        print("Indian Recipe recommender initialized successfully!")
        print(f"Trained on {len(self.recipes_df)} recipes with {len(self.all_ingredients)} unique ingredients.")

    def _preprocess_data(self):
        self.recipes_df['RecipeName'] = self.recipes_df['TranslatedRecipeName'].str.strip()
        # Use clean list for all matching
        self.recipes_df['CleanIngredientsList'] = self.recipes_df[self.clean_ingredient_col].apply(
            lambda x: [i.strip().lower() for i in x.split(',')] if pd.notna(x) else []
        )
        # For display, parse original ingredient text
        self.recipes_df['IngredientsList'] = self.recipes_df['TranslatedIngredients'].apply(self._parse_ingredients)
        self.recipes_df['IngredientsText'] = self.recipes_df['CleanIngredientsList'].apply(lambda x: ' '.join(x))
        # Parse instructions into steps
        self.recipes_df['InstructionsList'] = self.recipes_df['TranslatedInstructions'].apply(self._parse_instructions)

    def _parse_ingredients(self, ingredients_text):
        if pd.isna(ingredients_text):
            return []
        ingredients = [ing.strip() for ing in ingredients_text.split(',')]
        return ingredients

    def _parse_instructions(self, instructions_text):
        if pd.isna(instructions_text):
            return []
        steps = [step.strip() for step in re.split(r'\.\s+(?=[A-Z])|\n', instructions_text) if step.strip()]
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Indian Recipe Recommender System')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the Indian recipes CSV file')
    parser.add_argument('--top', type=int, default=5, help='Number of recommendations to show')
    return parser.parse_args()

def main():
    args = parse_arguments()
    try:
        recommender = IndianRecipeRecommender(args.dataset)
        print("\nWelcome to the Indian Recipe Recommender!")
        print("Enter your ingredients (separated by commas), or 'q' to quit")
        while True:
            try:
                user_input = input("\nEnter ingredients: ")
                if user_input.lower() == 'q':
                    print("Thank you for using the Indian Recipe Recommender. Goodbye!")
                    break
                ingredients = [item.strip() for item in user_input.split(',')]
                if not ingredients or all(len(i.strip()) == 0 for i in ingredients):
                    print("Please enter at least one ingredient.")
                    continue
                recommendations = recommender.recommend_recipes(ingredients, args.top)
                if recommendations:
                    print(f"\nFound {len(recommendations)} Indian recipes you can make:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"\n{i}. {rec['title']} (Confidence: {rec['confidence']}%)")
                        print(f"   Required ingredients: {rec['ingredients']}")
                        if rec['missing_ingredients']:
                            print(f"   Missing ingredients: {', '.join(rec['missing_ingredients'])}")
                        print("   Instructions:")
                        for step_num, step in enumerate(rec['instructions'], 1):
                            print(f"     {step_num}. {step}")
                else:
                    print("\nNo matching recipes found with those ingredients.")
                    print("Try adding more common Indian ingredients or check your spelling.")
            except KeyboardInterrupt:
                print("\nThank you for using the Indian Recipe Recommender. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

