import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data provided
data = {
  "Variation A": {
    "A": {"delta": 0.133721, "variance_change": 0.089186},
    "B": {"delta": -0.030725, "variance_change": 0.021423},
    "C": {"delta": -0.054971, "variance_change": 0.036397},
    "D": {"delta": -0.052550, "variance_change": 0.040661}
  },
  "Variation B": {
    "A": {"delta": -0.024159, "variance_change": 0.044397},
    "B": {"delta": 0.116892, "variance_change": 0.077462},
    "C": {"delta": -0.050021, "variance_change": 0.040859},
    "D": {"delta": -0.054428, "variance_change": 0.057030}
  },
  "Variation C": {
    "A": {"delta": -0.032297, "variance_change": 0.053077},
    "B": {"delta": -0.081020, "variance_change": 0.062260},
    "C": {"delta": 0.128057, "variance_change": 0.085682},
    "D": {"delta": -0.026903, "variance_change": 0.030636}
  },
  "Variation D": {
    "A": {"delta": -0.067237, "variance_change": 0.054606},
    "B": {"delta": -0.117229, "variance_change": 0.082215},
    "C": {"delta": -0.061546, "variance_change": 0.052550},
    "D": {"delta": 0.235362, "variance_change": 0.149622}
  }
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame({
    "Variation": [],
    "Category": [],
    "Delta": [],
    "Variance Change": []
})

for variation, categories in data.items():
    for category, values in categories.items():
        df = df._append({
            "Variation": variation,
            "Category": category,
            "Delta": values["delta"],
            "Variance Change": values["variance_change"]
        }, ignore_index=True)

# Create the barplot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Draw the barplot
barplot = sns.barplot(
    x="Category", y="Delta", hue="Variation",
    data=df, palette="muted", ci=None
)

# Add error bars manually with alignment
bar_width = 0.8 / len(df['Variation'].unique())  # Divide the total bar width by number of variations
for i, variation in enumerate(df['Variation'].unique()):
    var_df = df[df['Variation'] == variation]
    x = np.arange(len(var_df['Category'])) - 0.4 + bar_width / 2 + i * bar_width
    
    plt.errorbar(
        x=x,
        y=var_df['Delta'],
        yerr=var_df['Variance Change'],
        fmt='none',
        capsize=5,
        color='black'
    )

# Customize the plot
plt.title("Average Linear Probability Change for each Visual Bias")
plt.ylabel("Change in Token Linprob")
plt.xlabel("Bias Type")
plt.legend(title="Answer Choices", labels=["Choice A", "Choice B", "Choice C", "Choice D"])
plt.tight_layout()

# Show the plot
plt.show()
