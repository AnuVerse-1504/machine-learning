# Import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample dataset
data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']
})

print("Original Data:")
print(data)

# Create OneHotEncoder object
encoder = OneHotEncoder(sparse_output=False)

# Apply encoding
encoded = encoder.fit_transform(data[['Color']])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Color']))

print("\nEncoded Data:")
print(encoded_df)