import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data provided
data = {
  "Variation A": {
    "A": {"delta": 0.083587, "variance_change": 0.041156},
    "B": {"delta": -0.071263, "variance_change": 0.036518},
    "C": {"delta": -0.012299, "variance_change": 0.004475}
  },
  "Variation B": {
    "A": {"delta": -0.021157, "variance_change": 0.015725},
    "B": {"delta": 0.070528, "variance_change": 0.036301},
    "C": {"delta": -0.049325, "variance_change": 0.023025}
  },
  "Variation C": {
    "A": {"delta": -0.009708, "variance_change": 0.017343},
    "B": {"delta": -0.095299, "variance_change": 0.065771},
    "C": {"delta": 0.105004, "variance_change": 0.077551}
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
