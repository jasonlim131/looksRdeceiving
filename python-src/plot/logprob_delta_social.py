import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# TODO: store the data in a separate file and import it

# Data provided
data = {
    "Variation A": {
        "A": {"delta": 0.082926, "variance_change": 0.065265},
        "B": {"delta": -0.017214, "variance_change": 0.048603},
        "C": {"delta": -0.065705, "variance_change": 0.064767}
    },
    "Variation B": {
        "A": {"delta": -0.004211, "variance_change": 0.038393},
        "B": {"delta": 0.089379, "variance_change": 0.063094},
        "C": {"delta": -0.085181, "variance_change": 0.063769}
    },
    "Variation C": {
        "A": {"delta": -0.073175, "variance_change": 0.055146},
        "B": {"delta": -0.094715, "variance_change": 0.078693},
        "C": {"delta": 0.167894, "variance_change": 0.119649}
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
bar_width = 0.8 / len(df['Variation'].unique()) 
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
