import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# TODO: store the data in a separate file and import it

# New Data provided
data = {
  "Variation A": {
    "A": {"delta": 0.088718, "variance_change": 0.053075},
    "B": {"delta": 0.000498, "variance_change": 0.013683},
    "C": {"delta": -0.036028, "variance_change": 0.021498},
    "D": {"delta": -0.047124, "variance_change": 0.033000}
  },
  "Variation B": {
    "A": {"delta": -0.028804, "variance_change": 0.023531},
    "B": {"delta": 0.129737, "variance_change": 0.075132},
    "C": {"delta": -0.045442, "variance_change": 0.031393},
    "D": {"delta": -0.057361, "variance_change": 0.045900}
  },
  "Variation C": {
    "A": {"delta": -0.009011, "variance_change": 0.012552},
    "B": {"delta": -0.026379, "variance_change": 0.018694},
    "C": {"delta": 0.058318, "variance_change": 0.037541},
    "D": {"delta": -0.027061, "variance_change": 0.019414}
  },
  "Variation D": {
    "A": {"delta": -0.064358, "variance_change": 0.041644},
    "B": {"delta": -0.060623, "variance_change": 0.049766},
    "C": {"delta": -0.048612, "variance_change": 0.037652},
    "D": {"delta": 0.167436, "variance_change": 0.113425}
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
plt.legend(title="Answer Choices", labels=["Choice A", "Choice B", "Choice C", "Choice D"]
)
plt.tight_layout()

# Show the plot
plt.show()
