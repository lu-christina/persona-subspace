import pandas as pd

# Read the two CSV files
autointerp_df = pd.read_csv('0_personal_general_autointerp.csv')
general_df = pd.read_csv('2_personal_general.csv')

# Rename columns in general_df to add 'general_' prefix
general_df = general_df.rename(columns={
    'activation_mean': 'general_activation_mean',
    'activation_max': 'general_activation_max',
    'activation_min': 'general_activation_min'
})

# Select only the columns we need from each dataframe
general_cols = ['feature_id', 'general_activation_mean', 'general_activation_max', 'general_activation_min', 'num_prompts']
autointerp_cols = ['feature_id', 'personal_mean', 'personal_cohens_d', 'chat_desc', 'pt_desc', 'type', 'source', 'token', 'link', 'claude_completion', 'claude_desc', 'claude_type']

# Merge the dataframes on feature_id, using the order from 2_personal_general.csv
merged_df = pd.merge(
    general_df[general_cols],
    autointerp_df[autointerp_cols],
    on='feature_id',
    how='left'
)

# Reorder columns to match the specified order
final_columns = [
    'feature_id',
    'general_activation_mean',
    'general_activation_max', 
    'general_activation_min',
    'personal_mean',
    'personal_cohens_d',
    'chat_desc',
    'pt_desc',
    'type',
    'source',
    'token',
    'num_prompts',
    'link',
    'claude_completion',
    'claude_desc',
    'claude_type'
]

# Select and reorder columns
merged_df = merged_df[final_columns]

# Save to the specified filename
merged_df.to_csv('2_personal_general_desc.csv', index=False)

print(f"Merged file created with {len(merged_df)} rows")
print(f"Columns: {list(merged_df.columns)}")