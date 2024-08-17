import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data provided
data = {
    "Variation A": {
        "A": {"delta": -0.000284, "variance_change": 0.000335},
        "B": {"delta": -0.001185, "variance_change": 0.000105},
        "C": {"delta": 0.001016, "variance_change": 0.000388},
        "D": {"delta": 0.000453, "variance_change": 0.000170}
    },
    "Variation B": {
        "A": {"delta": -0.001543, "variance_change": 0.001896},
        "B": {"delta": 0.000036, "variance_change": 0.000122},
        "C": {"delta": -0.002842, "variance_change": 0.000628},
        "D": {"delta": 0.004349, "variance_change": 0.001706}
    },
    "Variation C": {
        "A": {"delta": -0.004433, "variance_change": 0.000559},
        "B": {"delta": 0.000132, "variance_change": 0.000151},
        "C": {"delta": 0.000675, "variance_change": 0.000401},
        "D": {"delta": 0.003625, "variance_change": 0.000428}
    },
    "Variation D": {
        "A": {"delta": -0.000036, "variance_change": 0.000077},
        "B": {"delta": 0.000064, "variance_change": 0.000046},
        "C": {"delta": -0.002998, "variance_change": 0.000221},
        "D": {"delta": 0.002970, "variance_change": 0.000322}
    }
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame({
    "Variation": [],
    "Choice": [],
    "Delta": [],
    "Variance Change": []
})

for variation, choices in data.items():
    for choice, values in choices.items():
        df = df._append({
            "Variation": variation,
            "Choice": choice,
            "Delta": values["delta"],
            "Variance Change": values["variance_change"]
        }, ignore_index=True)

# Create the barplot
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

# Draw the barplot
barplot = sns.barplot(
    x="Variation", y="Delta", hue="Choice",
    data=df, palette="muted", ci=None
)

# Add error bars manually
for i, choice in enumerate(df['Choice'].unique()):
    choice_data = df[df['Choice'] == choice]
    x = np.arange(len(choice_data['Variation']))
    plt.errorbar(
        x=x + i * 0.2 - 0.3,  # Adjust bar positions
        y=choice_data['Delta'],
        yerr=choice_data['Variance Change'],
        fmt='none',
        capsize=5,
        color='black'
    )

# Customize the plot
plt.title("Average Linear Probability Change for each Visual Bias")
plt.ylabel("Change in Token Linprob")
plt.xlabel("Variation")
plt.legend(title="Answer Choices")
plt.tight_layout()

# Show the plot
plt.show()