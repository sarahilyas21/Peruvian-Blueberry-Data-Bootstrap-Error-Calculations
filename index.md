# Error Calculations for Peruvian Bluberry Data: Price, Volume and Value

## Introduction

We are interested in calculating standard errors for quantities such as price, value and volume. 
A parametric approach in our case would be asuming that the data for a specific week follows (let's say) a normal distribution. Normal distribution is the most widely used distribution in statistics. However, we must check our data's distribution before we make such a claim.

A widely used test in Statistics to check for normality is called the Shapiro Wilk test, 
We can carry out this test on some rows of our data to see if pricing for any given week is normally distributed. 
Below, we carry out the test on row 47 of our data.


```python
#Import the necessary libraries
import pandas as pd
from scipy.stats import shapiro
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python

file_path = 'Price.xlsx'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

first_row = df.iloc[47,:] #Select the 47th row

# Perform the Shapiro-Wilk test
statistic, p_value = shapiro(first_row)

# Display the test result
print("Shapiro-Wilk Test Statistic:", statistic)
print("P-value:", p_value)

# Interpret the result
alpha = 0.05
if p_value > alpha:
    print("The sample looks normally distributed (fail to reject H0)")
else:
    print("The sample does not look normally distributed (reject H0)")


```

    Shapiro-Wilk Test Statistic: 0.7900190949440002
    P-value: 0.0052005513571202755
    The sample does not look normally distributed (reject H0)


According to our results, the data is not normally distributed. Therefore we follow a non-parametric approach for the calculation of the standard errors. A non parametric approach is one that does not assume an underlying distribution for the given data.

## Non-Parametric Approach - Bootstrapping

Bootstrapping is a resampling technique that can be used to estimate the sampling distribution of a statistic, such as the standard error, even when the underlying data is not normally distributed. This makes bootstrapping a versatile and robust method for estimating parameters.


Bootstrapping is a non-parametric method, meaning it does not rely on assumptions about the underlying distribution of the data. Instead, it directly uses the observed data to estimate the sampling distribution of a statistic. This makes it more robust and flexible, especially when the underlying distribution is unknown or not easily characterized. Bootstrapping involves randomly sampling from the observed data with replacement to create multiple bootstrap samples. This process effectively captures the variability and structure of the original data, allowing for more accurate estimation of parameters and uncertainty measures.


Overall, bootstrapping is a powerful and widely used technique for statistical inference and estimation, providing valuable insights even in cases where the data is not normally distributed.

## Moving Block Bootstrapping for Time Series Data

Moving block bootstrapping is particularly useful for time series data because it takes into account the temporal structure and dependencies present in the data. Time series data often exhibits temporal dependence, where the value of a data point is related to the values of previous data points. Moving block bootstrapping preserves this temporal dependence by sampling contiguous blocks of data, allowing the bootstrap samples to capture the autocorrelation structure of the original time series. It maintains the sequential ordering of observations in the time series. This is crucial for time series analysis, as the order of observations often carries important information about the underlying process.



## Moving Block Bootstrapping With Overlap

In bootstrapping with time series data, using an overlap can be beneficial for several reasons:

Preserving Temporal Dependence: Overlapping blocks allow for the preservation of temporal dependence in the resampled data. By including overlapping segments from adjacent blocks, the resampled data maintains some level of continuity and autocorrelation structure, which is essential for capturing the characteristics of the original time series.

Reduced Variance: Overlapping blocks can help reduce the variance of estimates derived from bootstrapping. By incorporating information from neighboring blocks, the resampled data may exhibit less variability, leading to more stable estimates of parameters and statistics.


## Error Calculations
I have used a modified formula for the calculation of the standard deviation. In place of mean, I have used the target value ( which is the last value in the respective row and the value that we are comparing each value in the past to)

$$
\text{Standard Deviation (}\sigma\text{)} = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \text{target value})^2}{n}}
$$

Once we have the standard deviation, we can calculate the standard error for each week by dividing the standard deviation with the square root of the block size.

$$
\text{Standard Error }{} = \left( \frac{\sigma}{\sqrt{n}} \right)  
$$

## Code Summary


First, we Initialize variables **block_size** and **overlap** to specify the size of blocks and the overlap between consecutive blocks for moving block bootstrapping.

Then we set the number of bootstrap samples to generate (**num_bootstrap_samples**) to 1000. This helps us ensure that our results will be precise. This step could be thought of as replicating an experiment several times to get the best results. 

We then define a function **calculate_standard_error** to calculate the standard error for each row using moving block bootstrapping with overlap.

This function iterates over each row of the DataFrame and performs the following steps:<br>
1.Divides the row into blocks and generates bootstrap samples with overlap.<br>
2.Calculates errors by subtracting the value of interest (from the "LastColumn" of the row) from the bootstrap samples.<br>
3.Separates positive and negative errors.<br>
4.Computes standard errors for positive and negative errors using the formula for standard deviation.<br>
5.Returns the positive and negative standard errors as a Pandas Series.<br>
6.Applies the calculate_standard_error function to each row of the DataFrame to compute the standard errors<br>
Both, the positive standard errors (positive_std_error) and negative standard errors (negative_std_error) are then printed and then the results are plotted. <br>
In the plot, positive standard errors are plotted above the original line while the negative standard errors are plotted below the original line. 

## Standard Error Calculations for Price (Using Moving Block Bootstrapping With Overlap)


```python
file_path = 'Price.xlsx'  # Read the Excel file containing the data

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

# Set option to display all rows of a DataFrame if it is printed
pd.set_option('display.max_rows', None)

df['LastColumn'] = df.iloc[:, -1]  # Extract the last column of the DataFrame as a new column named 'LastColumn'
block_size = 4  # Size of the blocks used in moving block bootstrapping
overlap = 3  # Size of the overlap between consecutive blocks

num_bootstrap_samples = 1000  # Number of bootstrap samples to generate
custom_labels = ['42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43','44','45','46','47','48','49']

```


```python
# Function to calculate standard error for each row using moving block bootstrapping with overlap
def calculate_standard_error(row):
    # Extract the value of interest from the row
    value_of_interest = row['LastColumn']
    
    # Calculate the number of blocks and initialize array to store bootstrap samples
    num_blocks = (len(row[:-1]) - block_size) // overlap + 1
    bootstrap_samples = np.zeros((num_bootstrap_samples, block_size))
    
    last_index = 0  # Track the last used index
    # Generate bootstrap samples using moving block bootstrapping with overlap
    for j in range(num_bootstrap_samples):
        start_index = last_index
        last_index += overlap  # Increment by the overlap size for the next block
        if last_index + block_size > len(row[:-1]):
            last_index = 0  # Wrap around if the next block goes beyond the array
        bootstrap_samples[j] = row[start_index:start_index + block_size].values
    
    # Flatten the bootstrap samples array
    bootstrap_samples = bootstrap_samples.reshape((num_bootstrap_samples, -1))
    
    # Calculate errors (deviation from value of interest)
    errors = value_of_interest - bootstrap_samples

    # Separate positive and negative errors
    positive_errors = errors[errors >= 0]
    
    negative_errors = errors[errors < 0]
    
    # Calculate standard error for positive and negative errors
    positive_std_error = np.sqrt((sum((positive_errors)**2) / len(positive_errors))) / np.sqrt(block_size) if len(positive_errors) > 0 else 0
    negative_std_error = np.sqrt((sum((negative_errors)**2) / len(negative_errors))) / np.sqrt(block_size) if len(negative_errors) > 0 else 0
   
    # Return standard errors as a Pandas Series
    return pd.Series({'Positive_Standard_Error': positive_std_error, 'Negative_Standard_Error': negative_std_error})

# Calculate standard error for each row using moving block bootstrapping with overlap
standard_errors = df.apply(calculate_standard_error, axis=1)

# Create DataFrame for standard errors
df_se = pd.DataFrame(standard_errors)

# Create DataFrame for custom labels
df_cl = pd.DataFrame(custom_labels, columns=['Week'])

# Merge custom labels DataFrame with standard  errors DataFrame
merged_df = pd.merge(df_cl, df_se, left_index=True, right_index=True)
merged_df.index.name = None  # Remove index name
(merged_df.head(62))

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Week</th>
      <th>Positive_Standard_Error</th>
      <th>Negative_Standard_Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>44</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>48</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>49</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>50</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>51</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>52</td>
      <td>0.039191</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9</td>
      <td>0.000000</td>
      <td>0.030000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10</td>
      <td>0.000000</td>
      <td>0.023884</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11</td>
      <td>0.000000</td>
      <td>0.210000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>12</td>
      <td>0.000000</td>
      <td>0.370000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>13</td>
      <td>0.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>14</td>
      <td>0.000000</td>
      <td>0.185000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>15</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>16</td>
      <td>0.000000</td>
      <td>0.005000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>17</td>
      <td>0.000000</td>
      <td>0.005000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18</td>
      <td>0.000000</td>
      <td>0.045000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>19</td>
      <td>0.043714</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>20</td>
      <td>0.060947</td>
      <td>0.015000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>21</td>
      <td>0.006250</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>22</td>
      <td>0.017912</td>
      <td>0.030000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>23</td>
      <td>0.022256</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>24</td>
      <td>0.105933</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>25</td>
      <td>0.185018</td>
      <td>0.005000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>26</td>
      <td>0.215433</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>27</td>
      <td>0.249321</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>28</td>
      <td>0.263391</td>
      <td>0.005000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>29</td>
      <td>0.311839</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>30</td>
      <td>0.442761</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>31</td>
      <td>0.526259</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>32</td>
      <td>0.514027</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>33</td>
      <td>0.502078</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>34</td>
      <td>0.505547</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>35</td>
      <td>0.547613</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>36</td>
      <td>0.675783</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>37</td>
      <td>0.744653</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>38</td>
      <td>0.683418</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>39</td>
      <td>0.807181</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50</th>
      <td>40</td>
      <td>0.821675</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>51</th>
      <td>41</td>
      <td>0.725644</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>52</th>
      <td>42</td>
      <td>0.891458</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>43</td>
      <td>1.071726</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>54</th>
      <td>44</td>
      <td>0.961428</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55</th>
      <td>45</td>
      <td>1.106961</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>56</th>
      <td>46</td>
      <td>1.224763</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>57</th>
      <td>47</td>
      <td>0.590099</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>58</th>
      <td>48</td>
      <td>0.084853</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>59</th>
      <td>49</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib.lines import Line2D
# Scatterplot with area chart and markers
fig, ax = plt.subplots(figsize=(18, 9))

# Plot the original line
ax.plot(range(len(df)), df['LastColumn'], color='#EA0000')

# Separate positive and negative standard errors
positive_errors = standard_errors['Positive_Standard_Error']
negative_errors = standard_errors['Negative_Standard_Error']

# Fill the area above the curve for positive errors
if not positive_errors.empty:
    fill=ax.fill_between(range(len(df)), df['LastColumn'], df['LastColumn'] + positive_errors, color='#EA0000', alpha=0.2, label='Positive Errors')
    fill=ax.fill_between(range(len(df)), df['LastColumn'], df['LastColumn'] + 2*positive_errors, color='#EA0000', alpha=0.15, label='Positive Errors')
# Fill the area below the curve for negative errors
if not negative_errors.empty:
    fill=ax.fill_between(range(len(df)), df['LastColumn'] - negative_errors.abs(), df['LastColumn'], color='#EA0000', alpha=0.3, label='Positive Errors')
    fill=ax.fill_between(range(len(df)), df['LastColumn'] - 2*negative_errors.abs(), df['LastColumn'], color='#EA0000', alpha=0.2, label='Positive Errors')
# Scatterplot with opaque circular markers
ax.scatter(range(len(df)), df['LastColumn'], color='#EA0000', s=38)

# Customize the plot
ax.set_xlabel('Weeks', fontsize=14, labelpad=22)  # Set x-axis label and adjust padding
ax.set_ylabel('Price (USD)', fontsize=14, labelpad=10)  # Set y-axis label and adjust padding
ax.yaxis.set_major_formatter('${:,.0f}'.format)  # Add a dollar sign to y-axis ticks

# Customize x-axis ticks
plt.xticks(range(len(custom_labels)), custom_labels)

# Set plot title
ax.set_title('Peru Blueberry Fresh Export Price By Partner | Cultivated Conventional', fontsize=16, pad=10)

# Add gridlines
ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, alpha=0.2)

# Set x-axis limit
ax.set_xlim(-1, len(df))

# Add legend
circle_line = Line2D([0], [0], color='red', marker='o',  markersize=6, markerfacecolor='red', alpha=0.7)
legend_handles = [  circle_line,fill]
legend_labels = [ 'Reported Price','Error']

ax.legend(handles=legend_handles, labels=legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

# Customize spine color
for spine in plt.gca().spines.values():
    spine.set_edgecolor('#d3d3d3')

# Save the figure
fig.savefig('Price_errors.png')

# Show the plot
plt.show()

```


    
![png](output_15_0.png)
    


## Standard Error Calculations for Value  (Using Moving Block Bootstrapping With Overlap)


```python

file_path = 'Value.xlsx'  # Read the excel file


# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)  # Read data into DataFrame


# Extract the last column as the column of interest
df['LastColumn'] = df.iloc[:, -1]

# Set parameters for moving block bootstrapping
block_size = 4  
overlap = 3     

# Set the number of bootstrap samples
num_bootstrap_samples = 4

# Define custom labels for the weeks
custom_labels = ['42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '1', '2', '3', '4', '5', '6', '7',
                 '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                 '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
                 '42', '43', '44', '45', '46', '47', '48', '49']

```


```python
# Function to calculate standard error for each row using moving block bootstrapping with overlap
def calculate_standard_error(row):
    value_of_interest = row['LastColumn']
    
    # Perform moving block bootstrapping with overlap
    num_blocks = (len(row[:-1]) - block_size) // overlap + 1
    bootstrap_samples = np.zeros((num_bootstrap_samples, block_size))
    
    last_index = 0  # Track the last used index
    for j in range(num_bootstrap_samples):
        start_index = last_index
        last_index += overlap  # Increment by the overlap size for the next block
        if last_index + block_size > len(row[:-1]):
            last_index = 0  # Wrap around if the next block goes beyond the array
        bootstrap_samples[j] = row[start_index:start_index + block_size].values
    
    # Flatten the bootstrap samples array
    bootstrap_samples = bootstrap_samples.reshape((num_bootstrap_samples, -1))
    
    # Calculate standard error
    errors = value_of_interest-bootstrap_samples
    
    positive_errors = errors[errors >= 0]  # Filter positive errors
    print(positive_errors)
    negative_errors = errors[errors < 0]    # Filter negative errors
    
    positive_std_error = np.sqrt((sum((positive_errors)**2)/len(positive_errors))) / np.sqrt(block_size) if len(positive_errors) > 0  else 0
    
    negative_std_error = np.sqrt((sum((negative_errors)**2)/len(negative_errors))) / np.sqrt(block_size) if len(negative_errors) > 0 else 0
    
    return pd.Series({'Positive_Standard_Error': positive_std_error, 'Negative_Standard_Error': negative_std_error})

# Calculate standard error for each row using moving block bootstrapping with overlap
standard_errors = df.apply(calculate_standard_error, axis=1)


# Create DataFrame for standard errors
df_se = pd.DataFrame(standard_errors)

# Create DataFrame for custom labels with column name 'Week'
df_cl = pd.DataFrame(custom_labels, columns=['Week'])

# Print a message indicating the purpose of the displayed results
print("Standard Error for each row using moving block bootstrapping with overlap:\n")

# Merge the custom labels DataFrame and standard errors DataFrame on their indices
merged_df = pd.merge(df_cl, df_se, left_index=True, right_index=True)

# Remove the index name
merged_df.index.name = None

# Display the merged DataFrame
(merged_df.head(62))

```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [5497187.24 5497187.24 5497187.24 5497187.24 5497187.24 5497187.24
     5497187.24 5497187.24 5497187.24 5497187.24 5497187.24 4122890.43
     4122890.43 2748593.62 1374296.81       0.  ]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [2452.99 2452.99 2452.99 2452.99 2452.99 2452.99    0.      0.      0.
        0.      0.      0.      0.      0.      0.      0.  ]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0.]
    [373.38   0.     0.     0.     0.     0.     0.     0.     0.     0.
       0.     0.     0.     0.  ]
    [0. 0. 0. 0. 0.]
    [58576.56 58576.56 58576.56 58576.56 58576.56 26598.88 26598.88     0.
         0.       0.       0.       0.       0.       0.       0.       0.  ]
    [60436.47 64962.74 64962.74 64962.74 64962.74 15451.62 15451.62     0.
         0.       0.       0.       0.  ]
    [15666.87     0.       0.       0.       0.       0.       0.       0.
         0.       0.       0.       0.       0.       0.  ]
    [46391.03 14563.34 14563.34 14563.34 14563.34 14563.34 14563.34     0.
         0.       0.       0.       0.  ]
    [26114.08 18024.83 18024.83 18024.83 18024.83 18024.83 11968.33 27953.88
     27953.88 22579.54     0.       0.  ]
    [219360.11 224756.87 213561.77 213561.77 213561.77 213561.77 207558.85
      65226.93  65226.93  52434.35      0.        0.        0.        0.
          0.  ]
    [426599.87 473376.01 469924.01 378519.28 378519.28 378519.28 193973.86
     217080.47 217080.47 200298.29      0.        0.        0.        0.
          0.  ]
    [766146.19 689006.82 624798.24 402125.01 402125.01 402125.01 305798.82
     227996.22 227996.22  89814.63      0.        0.        0.  ]
    [1373709.37 1289505.83 1096950.03  784130.39  784130.39  784130.39
      523791.56  336219.59  336219.59  326089.62       0.         0.
           0.         0.         0.         0.  ]
    [1586212.46 1478694.11 1254114.03  817985.22  817985.22  817985.22
      367093.79  245459.83  245459.83  211852.35       0.         0.  ]
    [2632202.55 2486193.84 1929341.01 1187741.67 1187741.67 1190908.37
      591965.32  244027.56  244027.56  192774.42   85177.     87429.51
       87429.51   27599.81       0.         0.  ]
    [4030457.07 3997503.02 3945873.85 2533620.35 2533620.35 2591394.24
     1310564.59  929059.82  929059.82  622840.63  117037.64   32456.38
       32456.38       0.         0.  ]
    [7082954.33 7053884.69 6827801.44 5319060.06 5319060.06 5065653.75
     2639363.74 1871539.62 1871539.62 1217803.46  224788.94  137883.48
      137883.48  143733.82       0.         0.  ]
    [7598351.32 7556358.35 7352144.75 6605770.17 6605770.17 6178594.43
     3454567.75 2354127.1  2354127.1  1339455.24  417010.4   292184.6
      292184.6   399568.12   72092.02       0.  ]
    [8106309.97       8056796.77       8056796.77       6916182.29
     6916182.29       6488801.68       4723055.27       3329025.18
     3329025.18       2166394.77       1114379.73999999  452760.13
      452760.13        246845.72         24197.66             0.        ]
    [8281261.44000001 8138180.44000001 8096948.54       7639435.65000001
     7639435.65000001 7601558.27       5798152.88       4661396.28
     4661396.28       3038444.29       1435920.54       1116527.13
     1116527.13        360987.72         56500.17             0.        ]
    [11186610.24  9026435.86  8269298.15  8256579.78  8256579.78  7940027.74
      7138836.93  6460720.55  6460720.55  5665002.19  3139010.23  2227878.01
      2227878.01  1131275.99    97699.68        0.  ]
    [24618072.62       13525203.75        9286553.25        8763806.97000001
      8763806.97000001  8608117.72000001  8249646.68        7847556.23
      7847556.23        6715231.12        4118270.42        3634993.46
      3634993.46        2191123.42         396543.07              0.        ]
    [27001926.93       24171004.88999999 10660220.79        7615169.98999999
      7615169.98999999  7401178.05999999  7338372.87        7232591.83
      7232591.83        7077326.08        4773665.2         3743284.88999999
      3743284.88999999  2298096.45         581246.38              0.        ]
    [33518404.25       25489967.33       12582662.         12582662.
      6527446.08        5789874.08        5982052.15000001  5982052.15000001
      5855774.28999999  5051887.86        4712073.81999999  4712073.81999999
      3794048.06        1021776.86              0.        ]
    [37449550.53       28786377.74       28786377.74        8738793.61
      5040047.91        4395674.17        4395674.17        4485559.91
      3979760.86        3693384.52        3693384.52        2795843.2
      1421619.65000001        0.        ]
    [33702610.49       33702610.49       27892974.46000001  9909926.42
      2179863.08        2179863.08        1681472.69000001  1633667.82000001
      1635093.84        1635093.84        1092170.16000001   838551.28
            0.        ]
    [39644498.30999999 31708516.68999999  5172073.83        5172073.83
      3628878.22999999   803541.31999999   278349.78999999   278349.78999999
       285627.72         272883.38999999        0.        ]
    [39679021.13       29397145.28999999 29397145.28999999 16630458.69
      2781951.52         759219.33         759219.33         268984.63
       223209.44              0.        ]
    [41730237.74999999 41730237.74999999 38784496.41999999 13833902.25
      3238000.27999999  3238000.27999999   775720.97999999   101935.64
            0.        ]
    [39094963.93       34001902.03       13812401.86       13812401.86
      3066980.81         674741.84999999        0.        ]
    [32449827.16       28698163.42999999 28698163.42999999 12377000.88
      2916196.88              0.        ]
    [28914909.36 28914909.36 24070520.62  8562518.4         0.  ]
    [15517731.52 10858795.75        0.  ]
    [2044647.90000001       0.        ]
    [0.]
    Standard Error for each row using moving block bootstrapping with overlap:
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Week</th>
      <th>Positive_Standard_Error</th>
      <th>Negative_Standard_Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>44</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>48</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>49</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>50</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>51</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>52</td>
      <td>2.423355e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4</td>
      <td>7.510717e+02</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9</td>
      <td>0.000000e+00</td>
      <td>46990.475000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10</td>
      <td>0.000000e+00</td>
      <td>24831.012831</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11</td>
      <td>0.000000e+00</td>
      <td>178421.605000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>12</td>
      <td>0.000000e+00</td>
      <td>161095.430000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>13</td>
      <td>0.000000e+00</td>
      <td>34555.325000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>14</td>
      <td>0.000000e+00</td>
      <td>57149.505000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>15</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>16</td>
      <td>0.000000e+00</td>
      <td>1020.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>17</td>
      <td>4.989500e+01</td>
      <td>813.540000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18</td>
      <td>0.000000e+00</td>
      <td>14095.805000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>19</td>
      <td>1.703446e+04</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>20</td>
      <td>2.092183e+04</td>
      <td>5745.640000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>21</td>
      <td>2.093574e+03</td>
      <td>828.480000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>22</td>
      <td>8.446736e+03</td>
      <td>12729.450000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>23</td>
      <td>9.706455e+03</td>
      <td>49.950000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>24</td>
      <td>7.476721e+04</td>
      <td>38.350000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>25</td>
      <td>1.431040e+05</td>
      <td>4068.430000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>26</td>
      <td>2.029940e+05</td>
      <td>12906.624589</td>
    </tr>
    <tr>
      <th>37</th>
      <td>27</td>
      <td>3.355165e+05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>28</td>
      <td>4.228585e+05</td>
      <td>4216.059066</td>
    </tr>
    <tr>
      <th>39</th>
      <td>29</td>
      <td>5.809706e+05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>30</td>
      <td>1.089311e+06</td>
      <td>2330.155000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>31</td>
      <td>1.953978e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>32</td>
      <td>2.234867e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>33</td>
      <td>2.450085e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>34</td>
      <td>2.698009e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>35</td>
      <td>3.216148e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>36</td>
      <td>4.656301e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>37</td>
      <td>5.394924e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>38</td>
      <td>6.267943e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>39</td>
      <td>7.645343e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50</th>
      <td>40</td>
      <td>7.808033e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>51</th>
      <td>41</td>
      <td>7.752921e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>52</th>
      <td>42</td>
      <td>9.471530e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>43</td>
      <td>1.201853e+07</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>54</th>
      <td>44</td>
      <td>1.048123e+07</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55</th>
      <td>45</td>
      <td>1.091989e+07</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>56</th>
      <td>46</td>
      <td>1.078158e+07</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>57</th>
      <td>47</td>
      <td>5.467430e+06</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>58</th>
      <td>48</td>
      <td>7.228922e+05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>59</th>
      <td>49</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Scatterplot with area chart and markers
fig, ax = plt.subplots(figsize=(18, 9))

# Plot the original line
ax.plot(range(len(df)), df['LastColumn'], color='#EA0000')

# Separate positive and negative standard errors
positive_errors = standard_errors['Positive_Standard_Error']
negative_errors = standard_errors['Negative_Standard_Error']

# Fill the area above the curve for positive errors
if not positive_errors.empty:
    fill=ax.fill_between(range(len(df)), df['LastColumn'], df['LastColumn'] + positive_errors, color='#EA0000', alpha=0.3)
    fill=ax.fill_between(range(len(df)), df['LastColumn'], df['LastColumn'] + 2*positive_errors, color='#EA0000', alpha=0.15, label='Positive Errors')
# Fill the area below the curve for negative errors
if not negative_errors.empty:
    fill=ax.fill_between(range(len(df)), df['LastColumn'] - negative_errors.abs(), df['LastColumn'], color='red', alpha=0.3)
    fill=ax.fill_between(range(len(df)), df['LastColumn'] - 2*negative_errors.abs(), df['LastColumn'], color='#EA0000', alpha=0.2, label='Positive Errors')
# Scatterplot with opaque circular markers
ax.scatter(range(len(df)), df['LastColumn'], color='#EA0000', s=38)

# Customize the plot
ax.set_xlabel('Weeks', fontsize=14, labelpad=22)
ax.set_ylabel('Value (USD)', fontsize=14, labelpad=10)
ax.yaxis.set_major_formatter('${:,.0f}'.format)  # Add a dollar sign to y-axis ticks

# Starting from 47 and ending at 49
plt.xticks(range(len(custom_labels)), custom_labels)

ax.set_title('Peru Blueberry Fresh Export Value By Partner | Cultivated Conventional', fontsize=16, pad=10)
ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, alpha=0.2)
ax.set_xlim(-1, len(df))

# Add legend
circle_line = Line2D([0], [0], color='red', marker='o',  markersize=6, markerfacecolor='red', alpha=0.7)
legend_handles = [  circle_line,fill]
legend_labels = [ 'Reported Value','Error']

ax.legend(handles=legend_handles, labels=legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

for spine in plt.gca().spines.values(): #Adjust the color for spines
    spine.set_edgecolor('#d3d3d3')
    
# Save the figure
fig.savefig('Value_errors1.png')

# Show the plot
plt.show()



```


    
![png](output_19_0.png)
    


## Standard Error Calculations for Volume  (Using Moving Block Bootstrapping With Overlap)


```python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Replace 'your_file_path.xlsx' with the actual path to your Excel file
file_path = 'Volume.xlsx'


# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)


df['LastColumn'] = df.iloc[:, -1]


```


```python
# Function to calculate standard error for each row using moving block bootstrapping with overlap
def calculate_standard_error(row):
    value_of_interest = row['LastColumn']
    
    # Perform moving block bootstrapping with overlap
    num_blocks = (len(row[:-1]) - block_size) // overlap + 1
    bootstrap_samples = np.zeros((num_bootstrap_samples, block_size))
    
    last_index = 0  # Track the last used index
    for j in range(num_bootstrap_samples):
        start_index = last_index
        last_index += overlap  # Increment by the overlap size for the next block
        if last_index + block_size > len(row[:-1]):
            last_index = 0  # Wrap around if the next block goes beyond the array
        bootstrap_samples[j] = row[start_index:start_index + block_size].values
    
    # Flatten the bootstrap samples array
    bootstrap_samples = bootstrap_samples.reshape((num_bootstrap_samples, -1))
 
    # Calculate standard error
    errors = value_of_interest-bootstrap_samples
     
    
    positive_errors = errors[errors >= 0]  # Filter positive errors

    negative_errors = errors[errors < 0]    # Filter negative errors
   
    #Standard error formula for Positive errors
    positive_std_error = np.sqrt((sum((positive_errors)**2)/len(positive_errors))) / np.sqrt(block_size) if len(positive_errors) > 0  else 0
    
    #Standard error formula for Negative errors
    negative_std_error = np.sqrt((sum((negative_errors)**2)/len(negative_errors))) / np.sqrt(block_size) if len(negative_errors) > 0 else 0
    
    #Return positive and negative standard errors when the calculate_standard_error function is called
    return pd.Series({'Positive_Standard_Error': positive_std_error, 'Negative_Standard_Error': negative_std_error})

# Call the calculate_standard_error function 
standard_errors = df.apply(calculate_standard_error, axis=1)

df_se = pd.DataFrame(standard_errors)
df_cl = pd.DataFrame(custom_labels,columns=['Week'])

# Display the results
print("Standard Error for each row using moving block bootstrapping with overlap:\n")

merged_df = pd.merge(df_cl, df_se, left_index=True, right_index=True)
merged_df.index.name = None

# Display the merged DataFrame

(merged_df.head(62))



```

    Standard Error for each row using moving block bootstrapping with overlap:
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Week</th>
      <th>Positive_Standard_Error</th>
      <th>Negative_Standard_Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>44</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>48</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>49</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>50</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>51</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>52</td>
      <td>685.292315</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>12</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>13</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>14</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>15</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>16</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>17</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>19</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>20</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>21</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>22</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>23</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>24</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>25</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>26</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>27</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>28</td>
      <td>0.000000</td>
      <td>0.000005</td>
    </tr>
    <tr>
      <th>39</th>
      <td>29</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>30</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>31</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>32</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>33</td>
      <td>0.000001</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>34</td>
      <td>0.983544</td>
      <td>2.889250</td>
    </tr>
    <tr>
      <th>45</th>
      <td>35</td>
      <td>0.991947</td>
      <td>14.480490</td>
    </tr>
    <tr>
      <th>46</th>
      <td>36</td>
      <td>0.000000</td>
      <td>43.624018</td>
    </tr>
    <tr>
      <th>47</th>
      <td>37</td>
      <td>0.000000</td>
      <td>24.329138</td>
    </tr>
    <tr>
      <th>48</th>
      <td>38</td>
      <td>71.955293</td>
      <td>40.584185</td>
    </tr>
    <tr>
      <th>49</th>
      <td>39</td>
      <td>3.250944</td>
      <td>52.200923</td>
    </tr>
    <tr>
      <th>50</th>
      <td>40</td>
      <td>1.338108</td>
      <td>14.166414</td>
    </tr>
    <tr>
      <th>51</th>
      <td>41</td>
      <td>2616.676053</td>
      <td>8.507557</td>
    </tr>
    <tr>
      <th>52</th>
      <td>42</td>
      <td>0.727256</td>
      <td>51.675162</td>
    </tr>
    <tr>
      <th>53</th>
      <td>43</td>
      <td>10.863382</td>
      <td>123.822454</td>
    </tr>
    <tr>
      <th>54</th>
      <td>44</td>
      <td>1531.564448</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55</th>
      <td>45</td>
      <td>23.955829</td>
      <td>38.939642</td>
    </tr>
    <tr>
      <th>56</th>
      <td>46</td>
      <td>18.067917</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>57</th>
      <td>47</td>
      <td>80.419596</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>58</th>
      <td>48</td>
      <td>28.767713</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>59</th>
      <td>49</td>
      <td>0.000000</td>
      <td>5.664740</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Scatterplot with area chart and markers
fig, ax = plt.subplots(figsize=(18, 9))

# Plot the original line
ax.plot(range(len(df)), df['LastColumn'], color='#EA0000')

# Separate positive and negative standard errors
positive_errors = standard_errors['Positive_Standard_Error']
negative_errors = standard_errors['Negative_Standard_Error']

# Fill the area above the curve for positive errors
if not positive_errors.empty:
    fill=ax.fill_between(range(len(df)), df['LastColumn'], df['LastColumn'] + positive_errors, color='#EA0000', alpha=0.3)
    fill=ax.fill_between(range(len(df)), df['LastColumn'], df['LastColumn'] + 2*positive_errors, color='#EA0000', alpha=0.15, label='Positive Errors')
# Fill the area below the curve for negative errors
if not negative_errors.empty:
    fill=ax.fill_between(range(len(df)), df['LastColumn'] - negative_errors.abs(), df['LastColumn'], color='#EA0000', alpha=0.3)
    fill=ax.fill_between(range(len(df)), df['LastColumn'] - 2*negative_errors.abs(), df['LastColumn'], color='#EA0000', alpha=0.2, label='Positive Errors')
# Scatterplot with opaque circular markers
ax.scatter(range(len(df)), df['LastColumn'], color='#EA0000', s=38)

# Customize the plot
ax.set_xlabel('Weeks', fontsize=14, labelpad=22)
ax.set_ylabel('Volume (KG)', fontsize=14, labelpad=10)
ax.yaxis.set_major_formatter('{:,.0f}M'.format)  # Add a dollar sign to y-axis ticks

# Add x axis ticks, starting from 47 and ending at 49
plt.xticks(range(len(custom_labels)), custom_labels)

ax.set_title('Peru Blueberry Fresh Export Volume By Partner | Cultivated Conventional', fontsize=16, pad=10)
ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, alpha=0.2)
ax.set_xlim(-1, len(df))

# Add legend
circle_line = Line2D([0], [0], color='red', marker='o',  markersize=6, markerfacecolor='red', alpha=0.7)
legend_handles = [  circle_line,fill]
legend_labels = [ 'Reported Volume','Error']

#Adjust the location of the legend
ax.legend(handles=legend_handles, labels=legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)


for spine in plt.gca().spines.values(): #Set the color for spines
    spine.set_edgecolor('#d3d3d3')
    
# Save the figure
fig.savefig('Volume_errors.png')

# Show the plot
plt.show()



```


    
![png](output_23_0.png)
    

