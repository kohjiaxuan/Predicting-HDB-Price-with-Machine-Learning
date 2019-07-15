# Predicting-HDB-Price-with-Machine-Learning
Data Project of Predicting HDB Resale Flat Prices with data cleaning, feature engineering and machine learning

More writeup would be done in the near future after the notebooks are tidied up!

Retrieved data from data.gov.sg on the prices of HDB Resale Flats from 2012 to 2019 and used it to build a strong predictive model for buyers and sellers of HDB flats to get a good pricing based on specifications of the house

# Important Features

Important features include: Floor Area of Flat, Neighborhood, Proximity to MRT, Storey Level of Unit, Year/Month purchased, Lease Remaining Time

# Data Cleaning and Transformation

Data cleaning was done to convert strings into integers or float values where necessary using pandas and lambda function. For example, STOREY 01-03 was converted to a mean float of 2. Hence, the variable became a float type. Similarly, year/month string in original data set was converted into float denoting the year. If it was in Jun 2018, the value would be 2018.5, i.e. halfway into the year.

For categorical variables like Neighborhood location (e.g. Bedok, Tampines, Jurong etc.), these were converted into new features/variables using pd.get dummies (where 0 means not in that neighborhood and 1 means apartment is in the neighborhood)

# Dealing with correlated features

Features/variables correlated to each other that could affect the final model were considered based on understanding and which one is more relevant. For example, the number of rooms in a flat is very correlated to floor area, but since the floor area is more precise (square metres) compared to (1-5 room flat), floor area was kept instead. Singaporeans would also know to tour the apartment and understand its size and take that into consideration more than the number of rooms of flat.

Lease remaining time was also calculated from the lease start year, so both features are correlated. In this case, we know that lease remaining time has a more direct influence on the price of flats so the former was used.

# Feature Normalization

Finally, all features were normalized by subtracting the mean and dividing by standard deviation. I found that this method has few outliers and is easier to use when subsequently testing new data. For example, a person living in Woodlands with floor size area of 120m2, we could easily normalize the data to test the results than use sklearn normalizer.

# Model Results

### Results
Random Forest (Project 7.) yielded the best results, with RMSE of 0.185, RMSLE 0.0237, R^2 0.97 on test data. Second place was neural network with 3 hidden layers (which was a bit too complex to justify the results), and third place was decision tree. Linear regression was the most intepretable followed by decision tree. Support Vector Regression loaded too slow (it's known to have high time complexity, but works very well for small datasets) and had poor results, while neural networks did not yield amazing results to justify using it since it is quite unintepretable.

### Decision Tree
The decision tree helped us to understand that the most important split is the floor area, and then the proximity to MRT, and subsequently some neighborhoods with extreme pricing (affordable vs pricey neighborhoods). It seemed to suggest that the features allowed a clear split for the HDB resale prices. However, since there were 51 features due to many neighborhoods, there might be overfitting from the decision tree splitting 51 times, so a random forest that selected 30-40 features in one stump acheieved the best results.

### Random Forest
This was by far the best model. With the random forest, only 1% of data had predicted HDB prices that were 100k difference from the actual buying/selling price. The mean difference between predicted and actual selling price was only 25k. Initially, the random forest did poorly when each stump (tree) was only 1-5 variables deep. I realized that is because there are some ultra key variables like floor size and proxmity to MRT that are way more important than other variables, so making the stump depth (number of variables) closer to the max 51 features but not totally allowed the majority of trees to choose these two variables along the way and build more robust trees. Conversely, if the features were more evenly spread out in terms of important, perhaps less features can be used for each stump instead.

### Linear Regression
The most intepretable model was the linear regression. With the linear regression, we could see how each unit of floor area increased the HDB price, and whether each neighborhood was expensive or cheaper. For example, neighborhoods like Queenstown and Bukit Timah caused flat prices to increase by more than 100k from the mean, but neighborhoods further from the city like Woodlands and Sembawang caused flat prices to fall by more than 100k from the mean. While there are more error than other models, it still performed a lot better than the baseline model of mean HDB price and helps us to understand the relative impact between variables.

### Neural Network
The neural network did not perform as well as the random forest in this case, even if there were 3 or 4 hidden layers with multiple nodes (e.g. 51,30,30,30,1) and maximal training iterations (n = 800-1200). I suspect that this could be because the neural network does not know that some features are really important compared to others to prioritize them, leading to difficulties in hitting a better local minima. Furthermore, neural network works better on data that have complex relationships with each other, and that is not really the case for this dataset where each feature can often mix and match quite well with other features without any relations. In any case, NN have poor interpretability and so I avoided it as the results are not as good as other traditional ML models.

### Support Vector Regression
The SVR loaded really slowly due to the large dataset and multiple features, but using a linear kernel to improve timing yielded poor results. Hence, other model alternatives were considered instead.

# Credits to other Git users
Many Git users wanted to use the Google API to get the coordinates of each neighborhood and get it's proximity to the MRT station, however there are some limitations to using the API (max request number capped) and lots of knowledge required to write the code to pass the query through. <b>I would like to give full credit to user leexa90 (https://github.com/leexa90/HDB-prices) for cracking this tough challenge and getting the MRT distance to each neighborhood which I used for this project.</b> From his dataset, I got the distances required and added it to my dataset, yielding significantly better results (reduction in 0.04 RMSE). This makes sense as housing prices are heavily determined by proximity to good transportation options, especially if a MRT station is close by, residents can easily get anywhere quickly, making the house more valuable. 

Also, thanks to all the other Git users doing this project that I have looked through for my inspiration, comparing model results and learning new techniques about Python, visualisation and sklearn.
