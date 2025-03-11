### Adding more features to the original dataset
#   (utilising a copy of our initial dataset)

library(tidyverse)   # For data manipulation and visualization
library(dplyr)       # For data wrangling
library(ggplot2)     # For visualizations
library(corrplot)    # For correlation analysis
library(skimr)       # For EDA summary
library(readxl)      # For reading Excel files

file_path <- "time_features_extracted_dataset.xlsx"
df_2 <- read_xlsx(file_path)  # operations to remove source and file columns, and convert label to factor done in console

### feature engineering
df_2 <- df_2 %>%
  mutate(skew_kurt_ratio = skewness / kurtosis)

df_2 <- df_2 %>%
  mutate(
    var = (rms^2 - median^2),  # Approximate variance from existing features
    crest_factor = max / rms,  # Crest Factor
    shape_factor = rms / abs(median),  # Shape Factor
    impulse_factor = max / abs(median),  # Impulse Factor
  )
glimpse(df_2)  # to verify if it worked
colSums(is.na(df_2)) # 25 NA values in some features

### same as before -> NA values are occuring because one of the features involved in operations is zero
#   to handle -> add a small constant to the features in denominator 

df_2 <- df_2 %>% select(-skew_kurt_ratio, -crest_factor, -shape_factor, -impulse_factor) %>%
  mutate(skew_kurt_ratio = skewness / (kurtosis + 1e-6),
         crest_factor = max / (rms + 1e-6),  # Crest Factor
         shape_factor = rms / (abs(median) + 1e-6),  # Shape Factor
         impulse_factor = max / (abs(median) + 1e-6),  # Impulse Factor
         )

df_2 <- df_2 %>%
  select(-label, everything(), label)  # to ensure label is last column


### Correlation analysis and dropping redundant features

# Compute correlation (excluding the 'label' column)
cor_matrix_v2 <- cor(df_2 %>% select(-label))

# Plot a correlation heatmap
corrplot::corrplot(cor_matrix_v2, 
                   method = "color",  # Use color shading
                   type = "full",  # Show full matrix instead of upper triangle
                   tl.col = "black",  # Set text color
                   tl.srt = 45,  # Rotate labels for better readability
                   diag = TRUE,  # Show diagonal values
                   addCoef.col = "black",  # Add correlation values in black text
                   number.cex = 0.75)  # Adjust text size

### from an advanced correlation analysis I identified which features are redundant

# engineering new features first

df_2 <- df_2 %>% mutate(skew_kurt_product = skewness * kurtosis,
                        std_min_ratio = abs(std / (min + 1e-6)),
                        min_ptp_ratio = min / (ptp + 1e-6))


# dropping some features 

df_2 <- df_2 %>% select(-mean, -std, -max, -impulse_factor, -skewness, -kurtosis)
glimpse(df_2)

df_3 <- df_2  # copying : df_3 will be used for RF and XGB

# Compute correlation (excluding the 'label' column) AGAIN!!
cor_matrix_v3 <- cor(df_2 %>% select(-label))

# Plot a correlation heatmap
corrplot::corrplot(cor_matrix_v3, 
                   method = "color",  # Use color shading
                   type = "full",  # Show full matrix instead of upper triangle
                   tl.col = "black",  # Set text color
                   tl.srt = 45,  # Rotate labels for better readability
                   diag = TRUE,  # Show diagonal values
                   addCoef.col = "black",  # Add correlation values in black text
                   number.cex = 0.75)  # Adjust text size

# ptp and rms are very highly correlated and from my last feature importance analysis for RF and XGB,
# rms was the least important feature so I'll drop it.

df_2 <- df_2 %>% select(-rms)
df_3 <- df_2 # copying again.....



### Outlier detection and handling (only for MLP, for RF I will use the dataset we have)


# Count outliers using IQR method
iqr_outliers <- df_2 %>%
  summarise(across(where(is.numeric), ~ sum(. < (quantile(., 0.25) - 1.5 * IQR(.)) | 
                                              . > (quantile(., 0.75) + 1.5 * IQR(.)))))

# Print number of outliers per feature
print(iqr_outliers) # many outliers -> need to be handled

# - option A : Winsorization -> to cap them at IQR*1.5 
# - option B : Robust scaling -> extreme values, even after scaling can adversely affect MLP

# Applying winsorization :

# Function to cap outliers at 1.5 * IQR range
cap_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  pmin(pmax(x, q1 - 1.5 * iqr), q3 + 1.5 * iqr)  # Capping values
}

# Apply Winsorization to all numeric features
df_2 <- df_2 %>%
  mutate(across(where(is.numeric), cap_outliers))

# Verify if outliers are reduced
iqr_outliers_after <- df_2 %>%
  summarise(across(where(is.numeric), ~ sum(. < (quantile(., 0.25) - 1.5 * IQR(.)) | 
                                              . > (quantile(., 0.75) + 1.5 * IQR(.)))))
print(iqr_outliers_after) # 0 remaining outliers

### Saving as csv

# Define file paths
rf_file_path <- "C:/Users/shaur/OneDrive/Desktop/TD_features_RF_V2.csv"
mlp_file_path <- "C:/Users/shaur/OneDrive/Desktop/TD_features_MLP_V2.csv"

# Save df_3 (for RF & XGB)
write.csv(df_3, rf_file_path, row.names = FALSE)

# Save df_2 (for MLP)
write.csv(df_2, mlp_file_path, row.names = FALSE)

# Confirm success
cat("✅ Datasets successfully saved to Desktop:\n",
    "- RF/XGB: ", rf_file_path, "\n",
    "- MLP: ", mlp_file_path, "\n")
















