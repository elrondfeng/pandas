

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
```


```python
input_file = 'purchase_data.json'
df = pd.read_json(input_file)
df.head()
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
      <th>Age</th>
      <th>Gender</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
      <th>SN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>Male</td>
      <td>165</td>
      <td>Bone Crushing Silver Skewer</td>
      <td>3.37</td>
      <td>Aelalis34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>Male</td>
      <td>119</td>
      <td>Stormbringer, Dark Blade of Ending Misery</td>
      <td>2.32</td>
      <td>Eolo46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>Male</td>
      <td>174</td>
      <td>Primitive Blade</td>
      <td>2.46</td>
      <td>Assastnya25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>Male</td>
      <td>92</td>
      <td>Final Critic</td>
      <td>1.36</td>
      <td>Pheusrical25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>Male</td>
      <td>63</td>
      <td>Stormfury Mace</td>
      <td>1.27</td>
      <td>Aela59</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.count()
```




    Age          780
    Gender       780
    Item ID      780
    Item Name    780
    Price        780
    SN           780
    dtype: int64




```python
df['Gender'].value_counts()
```




    Male                     633
    Female                   136
    Other / Non-Disclosed     11
    Name: Gender, dtype: int64




```python
print("total number of players is : " + str(df['SN'].nunique()))
```

    total number of players is : 573
    


```python
print("number of unique items is : " + str(df['Item ID'].value_counts().count()))
```

    number of unique items is : 183
    


```python
print("average purchase price is : "  +str(df['Price'].mean()))
```

    average purchase price is : 2.931192307692303
    


```python
print("total number of purchases is : "  +str(df['Item ID'].count()))
```

    total number of purchases is : 780
    


```python
print("total revenue is : "  +str(df['Price'].sum()))
```

    total revenue is : 2286.33
    


```python
# count of female, male and other players 
gender_count = df.groupby('Gender').SN.nunique()
gender_count
```




    Gender
    Female                   100
    Male                     465
    Other / Non-Disclosed      8
    Name: SN, dtype: int64




```python
# percentage of female, male and other players 
gender_count.apply(lambda x: x*100 / gender_count.sum())
```




    Gender
    Female                   17.452007
    Male                     81.151832
    Other / Non-Disclosed     1.396161
    Name: SN, dtype: float64




```python
#Purchasing Analysis (Gender)
#The below each broken by gender
#Purchase Count
#Average Purchase Price
#Total Purchase Value
#Normalized Totals
```


```python
grouped = df.groupby("Gender")
```


```python
#Purchase Count
grouped['Item ID'].count()
```




    Gender
    Female                   136
    Male                     633
    Other / Non-Disclosed     11
    Name: Item ID, dtype: int64




```python
#Average Purchase Price
grouped['Price'].mean()
```




    Gender
    Female                   2.815515
    Male                     2.950521
    Other / Non-Disclosed    3.249091
    Name: Price, dtype: float64




```python
#Total Purchase Value
grouped['Price'].sum()
```




    Gender
    Female                    382.91
    Male                     1867.68
    Other / Non-Disclosed      35.74
    Name: Price, dtype: float64




```python
#Normalized Totals
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(grouped)
#df_normalized = pd.DataFrame(x_scaled)
```


```python
#Age Demographics
#The below each broken into bins of 4 years (i.e. <10, 10-14, 15-19, etc.)
#Purchase Count
#Average Purchase Price
#Total Purchase Value
#Normalized Totals
```


```python
bins = np.arange(0,50,4)
bins
```




    array([ 0,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])




```python
#Purchase Count
df['age_bin'] = pd.cut(df["Age"],bins)
df.head()
df.groupby('age_bin')['Item ID'].count()
```




    age_bin
    (0, 4]        0
    (4, 8]       22
    (8, 12]      24
    (12, 16]     87
    (16, 20]    161
    (20, 24]    238
    (24, 28]    104
    (28, 32]     66
    (32, 36]     38
    (36, 40]     37
    (40, 44]      2
    (44, 48]      1
    Name: Item ID, dtype: int64




```python
#Average Purchase Price
df.groupby('age_bin')['Price'].mean()
```




    age_bin
    (0, 4]           NaN
    (4, 8]      2.788182
    (8, 12]     3.385417
    (12, 16]    2.745862
    (16, 20]    2.907019
    (20, 24]    2.924748
    (24, 28]    2.974712
    (28, 32]    3.061970
    (32, 36]    2.981053
    (36, 40]    2.901351
    (40, 44]    2.960000
    (44, 48]    2.720000
    Name: Price, dtype: float64




```python
#Total Purchase Value
df.groupby('age_bin')['Price'].sum()
```




    age_bin
    (0, 4]        0.00
    (4, 8]       61.34
    (8, 12]      81.25
    (12, 16]    238.89
    (16, 20]    468.03
    (20, 24]    696.09
    (24, 28]    309.37
    (28, 32]    202.09
    (32, 36]    113.28
    (36, 40]    107.35
    (40, 44]      5.92
    (44, 48]      2.72
    Name: Price, dtype: float64




```python
#Normalized Totals
```


```python
#Top Spenders
#Identify the the top 5 spenders in the game by total purchase value, then list (in a table):
#SN
#Purchase Count
#Average Purchase Price
#Totalpurchase value 
```


```python
df.head()
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
      <th>Age</th>
      <th>Gender</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
      <th>SN</th>
      <th>age_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>Male</td>
      <td>165</td>
      <td>Bone Crushing Silver Skewer</td>
      <td>3.37</td>
      <td>Aelalis34</td>
      <td>(36, 40]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>Male</td>
      <td>119</td>
      <td>Stormbringer, Dark Blade of Ending Misery</td>
      <td>2.32</td>
      <td>Eolo46</td>
      <td>(20, 24]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>Male</td>
      <td>174</td>
      <td>Primitive Blade</td>
      <td>2.46</td>
      <td>Assastnya25</td>
      <td>(32, 36]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>Male</td>
      <td>92</td>
      <td>Final Critic</td>
      <td>1.36</td>
      <td>Pheusrical25</td>
      <td>(20, 24]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>Male</td>
      <td>63</td>
      <td>Stormfury Mace</td>
      <td>1.27</td>
      <td>Aela59</td>
      <td>(20, 24]</td>
    </tr>
  </tbody>
</table>
</div>




```python
total = df.groupby('SN').Price.sum()
purchase_count = df.groupby('SN').Price.count()
Average = df.groupby('SN').Price.mean()

spenders = pd.DataFrame({'total':total,
            'purchase count': purchase_count,
            'Average Purchae Price' : Average    
})


spenders.nlargest(5,'total')
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
      <th>Average Purchae Price</th>
      <th>purchase count</th>
      <th>total</th>
    </tr>
    <tr>
      <th>SN</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Undirrala66</th>
      <td>3.412000</td>
      <td>5</td>
      <td>17.06</td>
    </tr>
    <tr>
      <th>Saedue76</th>
      <td>3.390000</td>
      <td>4</td>
      <td>13.56</td>
    </tr>
    <tr>
      <th>Mindimnya67</th>
      <td>3.185000</td>
      <td>4</td>
      <td>12.74</td>
    </tr>
    <tr>
      <th>Haellysu29</th>
      <td>4.243333</td>
      <td>3</td>
      <td>12.73</td>
    </tr>
    <tr>
      <th>Eoda93</th>
      <td>3.860000</td>
      <td>3</td>
      <td>11.58</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Most Popular Items
#Identify the 5 most popular items by purchase count, then list (in a table):
#Item ID
#Item Name
#Purchase Count
#Item Price
#Total Purchase Value
```


```python
df.head()
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
      <th>Age</th>
      <th>Gender</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
      <th>SN</th>
      <th>age_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>Male</td>
      <td>165</td>
      <td>Bone Crushing Silver Skewer</td>
      <td>3.37</td>
      <td>Aelalis34</td>
      <td>(36, 40]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>Male</td>
      <td>119</td>
      <td>Stormbringer, Dark Blade of Ending Misery</td>
      <td>2.32</td>
      <td>Eolo46</td>
      <td>(20, 24]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>Male</td>
      <td>174</td>
      <td>Primitive Blade</td>
      <td>2.46</td>
      <td>Assastnya25</td>
      <td>(32, 36]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>Male</td>
      <td>92</td>
      <td>Final Critic</td>
      <td>1.36</td>
      <td>Pheusrical25</td>
      <td>(20, 24]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>Male</td>
      <td>63</td>
      <td>Stormfury Mace</td>
      <td>1.27</td>
      <td>Aela59</td>
      <td>(20, 24]</td>
    </tr>
  </tbody>
</table>
</div>




```python

count = df.groupby("Item Name").Price.count()
price = df.groupby("Item Name").Price.mean()
total = df.groupby("Item Name").Price.sum()
item_id = df.groupby("Item Name")['Item ID'].mean().astype('int')

popular = pd.DataFrame({'Item id':item_id,
                       'Purchase Count' : count,
                       'Item Price' : price, 
                       'Total Purchase Value' : total})
popular.nlargest(5,'Purchase Count')


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
      <th>Item Price</th>
      <th>Item id</th>
      <th>Purchase Count</th>
      <th>Total Purchase Value</th>
    </tr>
    <tr>
      <th>Item Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Final Critic</th>
      <td>2.757143</td>
      <td>95</td>
      <td>14</td>
      <td>38.60</td>
    </tr>
    <tr>
      <th>Arcane Gem</th>
      <td>2.230000</td>
      <td>84</td>
      <td>11</td>
      <td>24.53</td>
    </tr>
    <tr>
      <th>Betrayal, Whisper of Grieving Widows</th>
      <td>2.350000</td>
      <td>39</td>
      <td>11</td>
      <td>25.85</td>
    </tr>
    <tr>
      <th>Stormcaller</th>
      <td>3.465000</td>
      <td>105</td>
      <td>10</td>
      <td>34.65</td>
    </tr>
    <tr>
      <th>Retribution Axe</th>
      <td>4.140000</td>
      <td>34</td>
      <td>9</td>
      <td>37.26</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Most Profitable Items
#Identify the 5 most profitable items by total purchase value, then list (in a table):
#Item ID
#Item Name
#Purchase Count
#Item Price
#Total Purchase Value
```


```python
popular.nlargest(5,'Total Purchase Value')
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
      <th>Item Price</th>
      <th>Item id</th>
      <th>Purchase Count</th>
      <th>Total Purchase Value</th>
    </tr>
    <tr>
      <th>Item Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Final Critic</th>
      <td>2.757143</td>
      <td>95</td>
      <td>14</td>
      <td>38.60</td>
    </tr>
    <tr>
      <th>Retribution Axe</th>
      <td>4.140000</td>
      <td>34</td>
      <td>9</td>
      <td>37.26</td>
    </tr>
    <tr>
      <th>Stormcaller</th>
      <td>3.465000</td>
      <td>105</td>
      <td>10</td>
      <td>34.65</td>
    </tr>
    <tr>
      <th>Spectral Diamond Doomblade</th>
      <td>4.250000</td>
      <td>115</td>
      <td>7</td>
      <td>29.75</td>
    </tr>
    <tr>
      <th>Orenmir</th>
      <td>4.950000</td>
      <td>32</td>
      <td>6</td>
      <td>29.70</td>
    </tr>
  </tbody>
</table>
</div>


