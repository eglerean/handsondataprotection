import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

# Load a specific sheet and specific columns
file_path='hodp2026_pseudonymised.csv'
df = pd.read_csv(file_path)

# Preview the DataFrame
print(df)

# drop the critical columns
# you need to do that yourself

# check xlsx file e
metadata = Metadata.detect_from_dataframe(data=df)

# try first
synthesizer = CTGANSynthesizer(metadata) # default epochs=300, batch_size = 500
                               
# Try the one below to improve results
# synthesizer = CTGANSynthesizer(metadata, epochs=3000, batch_size=200, pac=10, verbose = True, enforce_rounding=True, enforce_min_max_values=True)

synthesizer.fit(df)

synthetic_data = synthesizer.sample(num_rows=1000)
print(synthetic_data)
syn_file_path='synthetic_data.csv'
synthetic_data.to_csv(syn_file_path, index=False)

from sdv.evaluation.single_table import run_diagnostic

diagnostic_report = run_diagnostic(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata)

print(diagnostic_report)
