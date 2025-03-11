# Load libraries for data manipulation, visualization, and analysis
library(tidyverse)   # For data manipulation and visualization
library(dplyr)       # For data wrangling
library(ggplot2)     # For visualizations
library(corrplot)    # For correlation analysis
library(skimr)       # For EDA summary
library(readxl)      # For reading Excel files

file_path <- "time_features_extracted_dataset.xlsx"
df <- read_xlsx(file_path)

### identify missing values if any
colSums(is.na(df))
missmap(df)

### check structure
str(df) # label is categorical and needs to be converted to factor
df$label <- as.factor(df$label) 

### remove `source` and `file`
df %>% select(-source, -file)

### check class distribution
df %>%
  group_by(label) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) # dataset is balanced

### visualising class distribution
ggplot(df, aes(x = label, fill = factor(label))) +
  geom_bar(color = "black") +
  theme_minimal() +
  labs(title = "Class Distribution of Fault Types",
       x = "Fault Type",
       y = "Count",
       fill = "Fault Type") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

### identifying correlation between features

# Compute correlation (excluding the 'label' column)
cor_matrix <- cor(df %>% select(-label))

# Plot a correlation heatmap
corrplot::corrplot(cor_matrix, 
                   method = "color",  # Use color shading
                   type = "full",  # Show full matrix instead of upper triangle
                   tl.col = "black",  # Set text color
                   tl.srt = 45,  # Rotate labels for better readability
                   diag = TRUE,  # Show diagonal values
                   addCoef.col = "black",  # Add correlation values in black text
                   number.cex = 0.75)  # Adjust text size

### dropping redundant features (highly correlated => unsuitable for MLP)

df <- df %>%
  select(-mean, -max, -std)
# rationale : mean is nearly identical to median and rms
#             max and std are captured by ptp -> drop them

# Verify changes
glimpse(df)
# Observations from plotting correlation again :
# skewness and kurtosis are extremely negatively correlated
# min and ptp -> also high -ve, but keep because they capture different aspects of signal
# rms and ptp -> highly correlated but not perfect -> keep for now, for the same reasons as above

df <- df %>%
  mutate(skew_kurt_ratio = skewness / kurtosis) %>%
  select(-skewness, -kurtosis)  # using these two to create a new feature and dropping them after

# note for later : consider dropping ptp if MLP overfits

### Restructuring dataset so that target is the last column
df <- df %>%
  select(-label, everything(), label)

# Verify new column order
glimpse(df)
# check dataset for any missing values again
colSums(is.na(df)) # new feature has 25 NA values
                   # possibly because new feature was skewness/kurtosis
                   # if kurtosis is 0 this ratio would be NA
# verifying
df_copy <- read_xlsx("time_features_extracted_dataset.xlsx")
sum(df_copy$kurtosis == 0, na.rm = TRUE) # exactly 25 0s so assumption is correct.

# - Option A: Replace NA with 0 (simple but masks true values)
# - Option B: Replace NA with median (preserves distribution but introduces assumptions)
# - Option C (Chosen): Modify formula by adding a small constant to kurtosis

df <- df %>%
  select(-skew_kurt_ratio)  # Remove the current feature

# Add skewness and kurtosis back from the original copy
df <- df %>%
  mutate(skewness = df_copy$skewness, kurtosis = df_copy$kurtosis)

glimpse(df) # verify

# Recalculate `skew_kurt_ratio` while preventing division by zero
df <- df %>%
  mutate(skew_kurt_ratio = skewness / (kurtosis + 1e-6))

colSums(is.na(df)) # no more missing values

df <- df %>% select(-skewness, -kurtosis) # removing them again
glimpse(df)


### Outlier detection and handling

# Count outliers using IQR method
iqr_outliers <- df %>%
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
df <- df %>%
  mutate(across(where(is.numeric), cap_outliers))

# Verify if outliers are reduced
iqr_outliers_after <- df %>%
  summarise(across(where(is.numeric), ~ sum(. < (quantile(., 0.25) - 1.5 * IQR(.)) | 
                                              . > (quantile(., 0.75) + 1.5 * IQR(.)))))
print(iqr_outliers_after) # 0 remaining outliers

### feature scaling
# z-score standardization preferred for MLP
# reason : MLPs work best when features have a mean of 0 and a standard deviation of 1

# Standardize all numeric columns except 'label'
df_scaled <- df %>%
  mutate(across(where(is.numeric), ~ (. - mean(.)) / sd(.)))

# Verify scaled dataset
summary(df_scaled) # zcr's max is 3.44 which is >3 but likely not an issue
                   # it is within acceptable range


### Final Dataset Review

# Check dataset structure
glimpse(df_scaled) # the skew/kurt ratio is last column again
                   # reordering needs to be done to ensure target is last
df_scaled <- df_scaled %>%
  select(-label, everything(), label)

# Verify no missing values
colSums(is.na(df_scaled))

# Summary of the final dataset
summary(df_scaled)

### Saving the preprocessed dataset

# Save the dataset to a CSV file (accessible only within RStudio for now)
write.csv(df_scaled, "preprocessed_time_features_extracted_dataset.csv", row.names = FALSE)

# Verify it was saved
list.files(pattern = "preprocessed_time_features_extracted_dataset.csv")

### Exporting the dataset
# so i can model a XGBoost or RF Classifier in python

# Save the dataset to Desktop
write.csv(df, "C:\\Users\\shaur\\OneDrive\\Documents\\preprocessed_time_features_extracted_dataset.csv", row.names = FALSE)

# Verify if it's saved
list.files("C:\\Users\\shaur\\OneDrive\\Documents\\", pattern = "preprocessed_time_features_extracted_dataset.csv")

