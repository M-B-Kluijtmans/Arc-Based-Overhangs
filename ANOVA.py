import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.graphics.gofplots import qqplot
from IPython.display import clear_output

clear_output(wait=True)
plt.close('all')


data = {
    'A': [1, 1, 0.35, 0.35, 0.35, 0.35, 1, 1, 0.35, 1, 0.35, 1, 0.35, 1, 0.35, 1], #Threshold for small arcs
    'B': [2, 4, 4, 2, 4, 4, 2, 2, 2, 2, 4, 4, 2, 4, 2, 4], #Feedrate
    'C': [2, 2, 10, 10, 2, 10, 10, 10, 2, 2, 2, 10, 2, 2, 10, 10], #Minimum number of arcs per set
    'D': [0.75, 0.75, 0.75, 0.75, 0.75, 0.35, 0.75, 0.35, 0.35, 0.35, 0.35, 0.75, 0.75, 0.35, 0.35, 0.35], #Threshold to boundary line
    'Flatness': [0.15, 0.15, 0.17, 0.11, 0.22, 0.18, 0.12, 0.11, 0.13, 0.15, 0.28, 0.14, 0.14, 0.20, 0.12, 0.21], #Standard deviation of the geometry
    'Time': [15, 9, 8, 14, 8, 8, 15, 15, 16, 15, 9, 8, 15, 8, 15, 8], #Print time
    'Fill': [0.973, 0.977, 0.986, 0.990, 0.952, 0.991, 0.989, 0.994, 0.986, 0.981, 0.955, 0.980, 0.955, 0.955, 0.987, 0.988] #Fill percentage
    }
    
        
df = pd.DataFrame(data)      

   
# Fit the ANOVA model for Flatness
model_Flatness = ols('Flatness ~ (A + B + C + D)**2', data=df).fit()
anova_Flatness = sm.stats.anova_lm(model_Flatness, typ=2)
print("ANOVA results for Flatness:")
print(anova_Flatness)

# Fit the ANOVA model for Time
model_time = ols('Time ~ (A + B + C + D)**2', data=df).fit()
anova_time = sm.stats.anova_lm(model_time, typ=2)
print("ANOVA results for Time:")
print(anova_time)

# Fit the ANOVA model for Fill
model_fill = ols('Fill ~ (A + B + C + D)**2', data=df).fit()
anova_fill = sm.stats.anova_lm(model_fill, typ=2)
print("ANOVA results for Fill:")
print(anova_fill)

# ============== Flatness Plots ==============

# ======== Residual Plot for Flatness ========
residuals_dev = model_Flatness.resid
plt.figure(figsize=(8, 6))
plt.scatter(model_Flatness.fittedvalues, residuals_dev)
plt.axhline(0, linestyle='--', color='r')
plt.title('Residual Plot for Flatness')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# ======== Q-Q Plot for Flatness ========
qqplot(residuals_dev, line='s')
plt.title('Q-Q Plot for Flatness')
plt.show()

# ======== Interaction Plot (A and B on Flatness) ========
plt.figure(figsize=(8, 6))
interaction_plot(df['A'], df['B'], df['Flatness'])
plt.title('Interaction Plot: A and B on Flatness')
plt.show()

# ======== Interaction Plot (B and D on Flatness) ========
plt.figure(figsize=(8, 6))
interaction_plot(df['B'], df['D'], df['Flatness'])
plt.title('Interaction Plot: B and D on Flatness')
plt.show()

# ======== Boxplot of Flatness by Factor A ========
plt.figure(figsize=(8, 6))
sns.boxplot(x='A', y='Flatness', data=df)
plt.title('Boxplot of Flatness by Factor A')
plt.show()

# ======== Main Effects Plot for Flatness ========
plt.figure(figsize=(8, 6))
sns.pointplot(x='A', y='Flatness', data=df, color='blue', label='A')
sns.pointplot(x='B', y='Flatness', data=df, color='green', label='B')
sns.pointplot(x='C', y='Flatness', data=df, color='red', label='C')
sns.pointplot(x='D', y='Flatness', data=df, color='orange', label='D')
plt.title('Main Effects Plot for Flatness')
plt.xlabel('A, B, C, D')
# Modify x-axis labels
plt.xticks([0, 1], ['Min', 'Max'])
plt.legend(title='Factors')  # Adding legend with a title for the factors
plt.show()

# Group by each factor and calculate the mean of 'Flatness'
mean_A = df.groupby('A')['Flatness'].mean()
mean_B = df.groupby('B')['Flatness'].mean()
mean_C = df.groupby('C')['Flatness'].mean()
mean_D = df.groupby('D')['Flatness'].mean()

# Display the mean values
print("Mean Flatness for Factor A:")
print(mean_A)

print("\nMean Flatness for Factor B:")
print(mean_B)

print("\nMean Flatness for Factor C:")
print(mean_C)

print("\nMean Flatness for Factor D:")
print(mean_D)

# ============== Time Plots ==============

# ======== Residual Plot for Time ========
residuals_time = model_time.resid
plt.figure(figsize=(8, 6))
plt.scatter(model_time.fittedvalues, residuals_time)
plt.axhline(0, linestyle='--', color='r')
plt.title('Residual Plot for Time')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# ======== Q-Q Plot for Time ========
qqplot(residuals_time, line='s')
plt.title('Q-Q Plot for Time')
plt.show()

# ======== Interaction Plot (A and B on Time) ========
plt.figure(figsize=(8, 6))
interaction_plot(df['A'], df['D'], df['Time'])
plt.title('Interaction Plot: A and D on Time')
plt.show()


# ======== Boxplot of Time by Factor A ========
plt.figure(figsize=(8, 6))
sns.boxplot(x='A', y='Time', data=df)
plt.title('Boxplot of Time by Factor A')
plt.show()

# ======== Main Effects Plot for Time ========
plt.figure(figsize=(8, 6))
sns.pointplot(x='A', y='Time', data=df, color='blue', label='A')
sns.pointplot(x='B', y='Time', data=df, color='green', label='B')
sns.pointplot(x='C', y='Time', data=df, color='red', label='C')
sns.pointplot(x='D', y='Time', data=df, color='orange', label='D')
plt.title('Main Effects Plot for Print Time')
plt.xlabel('A, B, C, D')
plt.xticks([0, 1], ['Min', 'Max'])
plt.legend(title='Factors')
plt.show()

# Group by each factor and calculate the mean of 'Time'
mean_A = df.groupby('A')['Time'].mean()
mean_B = df.groupby('B')['Time'].mean()
mean_C = df.groupby('C')['Time'].mean()
mean_D = df.groupby('D')['Time'].mean()

# Display the mean values
print("Mean Time for Factor A:")
print(mean_A)

print("\nMean Time for Factor B:")
print(mean_B)

print("\nMean Time for Factor C:")
print(mean_C)

print("\nMean Time for Factor D:")
print(mean_D)


# ============== Fill Plots ==============

# ======== Residual Plot for Fill ========
residuals_fill = model_fill.resid
plt.figure(figsize=(8, 6))
plt.scatter(model_fill.fittedvalues, residuals_fill)
plt.axhline(0, linestyle='--', color='r')
plt.title('Residual Plot for Fill')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# ======== Q-Q Plot for Fill ========
qqplot(residuals_fill, line='s')
plt.title('Q-Q Plot for Fill')
plt.show()


# ======== Interaction Plot (B and D on Fill) ========
plt.figure(figsize=(8, 6))
interaction_plot(df['B'], df['D'], df['Fill'])
plt.title('Interaction Plot: B and D on Fill')
plt.show()

# ======== Boxplot of Fill by Factor A ========
plt.figure(figsize=(8, 6))
sns.boxplot(x='A', y='Fill', data=df)
plt.title('Boxplot of Fill by Factor A')
plt.show()

# ======== Main Effects Plot for Fill ========
plt.figure(figsize=(8, 6))
sns.pointplot(x='A', y='Fill', data=df, color='blue', label='A')
sns.pointplot(x='B', y='Fill', data=df, color='green', label='B')
sns.pointplot(x='C', y='Fill', data=df, color='red', label='C')
sns.pointplot(x='D', y='Fill', data=df, color='orange', label='D')
plt.title('Main Effects Plot for Fill Percentage')
plt.xlabel('A, B, C, D')
plt.xticks([0, 1], ['Min', 'Max'])
plt.legend(title='Factors')
plt.show()

# Group by each factor and calculate the mean of 'Fill'
mean_A = df.groupby('A')['Fill'].mean()
mean_B = df.groupby('B')['Fill'].mean()
mean_C = df.groupby('C')['Fill'].mean()
mean_D = df.groupby('D')['Fill'].mean()

# Display the mean values
print("Mean Fill for Factor A:")
print(mean_A)

print("\nMean Fill for Factor B:")
print(mean_B)

print("\nMean Fill for Factor C:")
print(mean_C)

print("\nMean Fill for Factor D:")
print(mean_D)

