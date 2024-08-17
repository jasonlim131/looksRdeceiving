import numpy as np
import pandas as pd
from scipy import stats

# reference_A_neutral_vmmlu = np.array([17.1806, 17.1806, 17.1806, 17.1806])
# optionA_neutral_vmmlu = np.array([47.2373, 8.3422, 7.5936, 8.5864])

# reference_B_neutral_vmmlu = np.array([18.6123, 18.6123, 18.6123, 18.6123])
# optionB_neutral_vmmlu = np.array([8.3422, 55.0802, 8.2353, 9.4241])

# reference_C_neutral_vmmlu = np.array([27.7533, 27.7533, 27.7533, 27.7533])
# optionC_neutral_vmmlu = np.array([16.2514, 16.0428, 72.7273, 9.3194])

# reference_D_neutral_vmmlu = np.array([36.4537, 36.4537, 36.4537, 36.4537])
# optionD_neutral_vmmlu = np.array([19.5016, 20.5348, 11.4439, 72.6702])

# # Neutral_Blue_Centered_Social_IQA
# reference_A_neutral_blue = np.array([25.3854, 25.3854, 25.3854])
# optionA_neutral_blue = np.array([55.2605, 12.6789, 14.7239])

# reference_B_neutral_blue = np.array([28.6742, 28.6742, 28.6742])
# optionB_neutral_blue = np.array([23.2891, 71.5746, 16.3599])

# reference_C_neutral_blue = np.array([45.9404, 45.9404, 45.9404])
# optionC_neutral_blue = np.array([21.4505, 15.7464, 68.9162])

# # Neutral_Social_IQA
# reference_A_neutral_social = np.array([21.1635, 21.1635, 21.1635])
# optionA_neutral_social = np.array([43.5743, 8.9089, 8.8176])

# reference_B_neutral_social = np.array([32.7984, 32.7984, 32.7984])
# optionB_neutral_social = np.array([28.5141, 74.2743, 15.6313])

# reference_C_neutral_social = np.array([46.0381, 46.0381, 46.0381])
# optionC_neutral_social = np.array([27.9116, 16.8168, 75.5511])


## gpt-4o-mini below

# Neutral_vmmlu
# reference_A_neutral_vmmlu = np.array([18.9583, 18.9583, 18.9583, 18.9583])
# optionA_neutral_vmmlu = np.array([32.5000, 18.4375, 16.0417, 15.0156])

# reference_B_neutral_vmmlu = np.array([34.5833, 34.5833, 34.5833, 34.5833])
# optionB_neutral_vmmlu = np.array([32.2917, 45.7292, 26.6667, 21.4807])

# reference_C_neutral_vmmlu = np.array([28.7500, 28.7500, 28.7500, 28.7500])
# optionC_neutral_vmmlu = np.array([22.9167, 24.0625, 42.5000, 22.2106])

# reference_D_neutral_vmmlu = np.array([17.7083, 17.7083, 17.7083, 17.7083])
# optionD_neutral_vmmlu = np.array([12.2917, 11.7708, 14.7917, 41.2930])

# # Neutral_Blue_Centered_Social_IQA
# reference_A_neutral_blue = np.array([30.0000, 30.0000, 30.0000])
# optionA_neutral_blue = np.array([38.0000, 26.6667, 30.0000])

# reference_B_neutral_blue = np.array([33.3333, 33.3333, 33.3333])
# optionB_neutral_blue = np.array([25.3333, 39.6667, 20.0000])

# reference_C_neutral_blue = np.array([36.6667, 36.6667, 36.6667])
# optionC_neutral_blue = np.array([36.6667, 33.6667, 50.0000])

# # Neutral_Social_IQA_vanilla
# reference_A_neutral_social = np.array([25.7000, 25.7000, 25.7000])
# optionA_neutral_social = np.array([34.2000, 26.0000, 19.1000])

# reference_B_neutral_social = np.array([35.0000, 35.0000, 35.0000])
# optionB_neutral_social = np.array([33.0000, 44.0000, 24.9000])

# reference_C_neutral_social = np.array([39.3000, 39.3000, 39.3000])
# optionC_neutral_social = np.array([32.8000, 30.0000, 56.0000])


## haiku below

# Option A: Compare A vs. B, C, D
reference_A_neutral_vmmlu = np.array([14.0, 14.0, 14.0, 14.0])
optionA_neutral_vmmlu = np.array([18.0, 24.0, 25.0, 28.0])

# Option B: Compare B vs. A, C, D
reference_B_neutral_vmmlu = np.array([24.0, 24.0, 24.0, 24.0])
optionB_neutral_vmmlu = np.array([14.0, 30.0, 25.0, 28.0])

# Option C: Compare C vs. A, B, D
reference_C_neutral_vmmlu = np.array([25.0, 25.0, 25.0, 25.0])
optionC_neutral_vmmlu = np.array([14.0, 24.0, 45.0, 28.0])

# Option D: Compare D vs. A, B, C
reference_D_neutral_vmmlu = np.array([28.0, 28.0, 28.0, 28.0])
optionD_neutral_vmmlu = np.array([14.0, 24.0, 25.0, 49.0])



# Neutral_Blue_Centered_Social_IQA

# Option A: Compare A vs. B, C
reference_A_neutral_blue = np.array([23.0, 23.0, 23.0])
optionA_neutral_blue = np.array([29.0, 30.0, 47.0])

# Option B: Compare B vs. A, C
reference_B_neutral_blue = np.array([30.0, 30.0, 30.0])
optionB_neutral_blue = np.array([22.0, 37.0, 47.0])

# Option C: Compare C vs. A, B
reference_C_neutral_blue = np.array([47.0, 47.0, 47.0])
optionC_neutral_blue = np.array([16.0, 22.0, 62.0])


# Neutral_Social_IQA

# Option A: Compare A vs. B, C
reference_A_neutral_social = np.array([24.0, 24.0, 24.0])
optionA_neutral_social = np.array([66.0, 17.0, 17.0])

# Option B: Compare B vs. A, C
reference_B_neutral_social = np.array([29.0, 29.0, 29.0])
optionB_neutral_social = np.array([3.0, 94.0, 3.0])

# Option C: Compare C vs. A, B
reference_C_neutral_social = np.array([47.0, 47.0, 47.0])
optionC_neutral_social = np.array([0.0, 0.0, 100.0])




# Option A
diff_A_vmmlu = optionA_neutral_vmmlu[0] - reference_A_neutral_vmmlu[0]
other_diffs_A_vmmlu = optionA_neutral_vmmlu[1:] - reference_A_neutral_vmmlu[1:]

# Option B
diff_B_vmmlu = optionB_neutral_vmmlu[1] - reference_B_neutral_vmmlu[1]
other_diffs_B_vmmlu = optionB_neutral_vmmlu[[0, 2, 3]] - reference_B_neutral_vmmlu[[0, 2, 3]]

# Option C
diff_C_vmmlu = optionC_neutral_vmmlu[2] - reference_C_neutral_vmmlu[2]
other_diffs_C_vmmlu = optionC_neutral_vmmlu[[0, 1, 3]] - reference_C_neutral_vmmlu[[0, 1, 3]]

# Option D
diff_D_vmmlu = optionD_neutral_vmmlu[3] - reference_D_neutral_vmmlu[3]
other_diffs_D_vmmlu = optionD_neutral_vmmlu[[0, 1, 2]] - reference_D_neutral_vmmlu[[0, 1, 2]]

# Perform paired t-tests
p_value_A_vmmlu_corrected = stats.ttest_1samp(other_diffs_A_vmmlu, diff_A_vmmlu).pvalue
p_value_B_vmmlu_corrected = stats.ttest_1samp(other_diffs_B_vmmlu, diff_B_vmmlu).pvalue
p_value_C_vmmlu_corrected = stats.ttest_1samp(other_diffs_C_vmmlu, diff_C_vmmlu).pvalue
p_value_D_vmmlu_corrected = stats.ttest_1samp(other_diffs_D_vmmlu, diff_D_vmmlu).pvalue

# For Blue-Centered Social IQA:
# Option A
diff_A_blue = optionA_neutral_blue[0] - reference_A_neutral_blue[0]
other_diffs_A_blue = optionA_neutral_blue[1:] - reference_A_neutral_blue[1:]

# Option B
diff_B_blue = optionB_neutral_blue[1] - reference_B_neutral_blue[1]
other_diffs_B_blue = optionB_neutral_blue[[0, 2]] - reference_B_neutral_blue[[0, 2]]

# Option C
diff_C_blue = optionC_neutral_blue[2] - reference_C_neutral_blue[2]
other_diffs_C_blue = optionC_neutral_blue[[0, 1]] - reference_C_neutral_blue[[0, 1]]

# Perform paired t-tests
p_value_A_blue_corrected = stats.ttest_1samp(other_diffs_A_blue, diff_A_blue).pvalue
p_value_B_blue_corrected = stats.ttest_1samp(other_diffs_B_blue, diff_B_blue).pvalue
p_value_C_blue_corrected = stats.ttest_1samp(other_diffs_C_blue, diff_C_blue).pvalue

# For Social IQA:
# Option A
diff_A_social = optionA_neutral_social[0] - reference_A_neutral_social[0]
other_diffs_A_social = optionA_neutral_social[1:] - reference_A_neutral_social[1:]

# Option B
diff_B_social = optionB_neutral_social[1] - reference_B_neutral_social[1]
other_diffs_B_social = optionB_neutral_social[[0, 2]] - reference_B_neutral_social[[0, 2]]

# Option C
diff_C_social = optionC_neutral_social[2] - reference_C_neutral_social[2]
other_diffs_C_social = optionC_neutral_social[[0, 1]] - reference_C_neutral_social[[0, 1]]

# Perform paired t-tests
p_value_A_social_corrected = stats.ttest_1samp(other_diffs_A_social, diff_A_social).pvalue
p_value_B_social_corrected = stats.ttest_1samp(other_diffs_B_social, diff_B_social).pvalue
p_value_C_social_corrected = stats.ttest_1samp(other_diffs_C_social, diff_C_social).pvalue

# Compile p-values into a readable format
p_values_summary_corrected = {
    "vMMLU": {
        "Option A": p_value_A_vmmlu_corrected,
        "Option B": p_value_B_vmmlu_corrected,
        "Option C": p_value_C_vmmlu_corrected,
        "Option D": p_value_D_vmmlu_corrected,
    },
    "Blue_Centered_Social_IQA": {
        "Option A": p_value_A_blue_corrected,
        "Option B": p_value_B_blue_corrected,
        "Option C": p_value_C_blue_corrected,
    },
    "Social_IQA": {
        "Option A": p_value_A_social_corrected,
        "Option B": p_value_B_social_corrected,
        "Option C": p_value_C_social_corrected,
    }
}

print(p_values_summary_corrected)
