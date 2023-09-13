# Finding the right place

At the moment there are more than 15.000 houses and apartments for sale in the small and beatiful city of Balneário Camboriú - Brazil, with an area of just 45 km², it means there are many options independent of where you want to live, near the Interpraias Park or near the Big Wheel.

![GasparRocha @ [pixabay.com](http://pixabay.com/)](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled.png)

GasparRocha @ [pixabay.com](http://pixabay.com/)

We all know that when overwhelmed with options we tend to not be able to choose from one, so to help out a friend who was searching for a place to buy in this city, I decided to analyse and sort all the options he had and find one that wasn’t overpriced, or even better, underpriced.

For this I decided to use statiscal analysis and ML models to predict the price considering all other houses, then find the ones that the real value is below the predicted price and have these as his choices.

# Collecting the data

To begin this search I’ve scrapped website’s listings and stored all the data in a Relational Database (MySQL). On the first day alone there were 9616 scrapped listing, this would grow to 15469 in another 20 days.

![Website scrapping in progress](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled_1(1).gif)

Website scrapping in progress

<aside>
💡 Check the code!
[https://github.com/krugergui/house-prices-balneario-camboriu/blob/main/scrape_data.ipynb](https://github.com/krugergui/house-prices-balneario-camboriu/blob/main/scrape_data.ipynb)

In this code you’ll find:
🐍 Python - Web scrapping with Selenium in Chrome Browser and JavaScript injection
🗃 SQL Database - Data retrieval from different tables, unification, update and alteration (new columns are inserted)

</aside>

The listings included the address, description, price, area, number of bedrooms and bathrooms, and sometimes a condominium fee and floor. There was also a huge list of amenities, in total 167 different types, I created  then 2 SQL tables inside the main Database, one for the listings and one for the amenities:

![Schema of the Database](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%201.png)

Schema of the Database

# Exploring the data

<aside>
💡 Look at the code!
[https://github.com/krugergui/house-prices-balneario-camboriu/blob/main/data_exploration.ipynb](https://github.com/krugergui/house-prices-balneario-camboriu/blob/main/data_exploration.ipynb)

In this code you’ll find:
🐍 Python
- Data manipulation with Pandas
- Data visualization with Matplotlib and Seaborn
- Statistics with Scipy
- Machine Learning with SKlearn

🗃 SQL Database - Data retrieval

</aside>

Using Pandas and **Matplotlib** we can take a quick look into what we are our pool of options:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%202.png)

Our price range is huge, on the right we can see the outliers as the dots, the cheapest option is R$250K, while on the top we can see our most expensive place at a staggering R$55M, 220 times more expensive than the cheapest one.

Analysing this boxplot (right) and the histogram on the top left we can take a look at all the prices, they are heavily concentrated below the R$5M mark (more than 83% of all entries).

On the middle of the left graphs we can take a look at the most expensive options, there are 74 of them above R$20M.

And in the bottom we see least expensive options, there are quite a few choices here, my friends stipulated around the R$1M range, around that mark, from R$800K to R$1.2M there are 766 options in our training set.

# Cleaning and tidying up the data

### Duplicates

After scraping up the data we have to clean it before analysing it, even though each entry was unique, sometimes the same place was put up several times, this can be because of several units of the same building for sale, or the same place was put into the websites system more than once, so we deleted some duplicated places.

## Creating new features

In the title of each entry there was the type house for sale, extracting this from the string results the following types for sale:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%203.png)

The first three are Apartment, House and Penthouse, respectively, after that we have commercial spots and free lots, which we are not interested in, so we’ll exclude those. Gated Community Houses (Casa de Condomínio) will also be included in the research.

This left us with this distribution:

![Count of every type of entries for sale](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%204.png)

Count of every type of entries for sale

### District

In the address there is also the district which the place finds itself in, to find out which are the districts from Balneário Camboriú I retrieved a table from Wikipedia. After the first run it was evident that there was an unofficial district - Barra Sul - which had very high prices.

![Count of every entry by district](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%205.png)

Count of every entry by district

Below we can see the price distribution among these districts:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%206.png)

Centro (City Center) has the higher range - as it has most of the entries, while Barra Sul has a higher average than the others. Vila Real has the cheapest entries, as it is located behind the highway far away from the beach, making it undesirable for most who come here looking to enjoy the sea.

![Google maps](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%207.png)

Google maps

# Reading up the data for the models

To have a better precision with the ML models a few things were checked on the data:

### Outliers

Outliers can heavily influence the outcomes of the prediction, a few causes for outliers are:

- Wrongly inputted/collected data
- Wrong unit of measure
- True outliers which should be analysed individually if they should be in the training data

This analysis was carried out in a few different ways, the first was to analyse the boxplots:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%208.png)

Here in the area box plot on the left we see that the top data point is really standing out, analysing the entry we see that it was an empty lot that was wrongly inputted as an apartment, so it will be removed.

On the lower range there were some outliers as well, taking a closer look at them we saw many values in the 10m², this is obviously impossible for an apartment, after cross-referencing with the websites listings it became clear that, when no value was given, the listing came with 10m² as default. These listing will also not be included in the training data.

To set a numerical value in what consists an outlier, rules was set according to literature. It is the first quantile (Q25) minus 1,5 times the inter quantile range, thus removing 44 of the entries.

On the price box plot below we also see a very strong outlier:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%209.png)

With a price of R$55M it is 70% more expensive than the second one, but this wasn’t a mistake, it is really a unique condo, so it will be included in the training data.

The same analysis was carried out on all numerical features.

### Normality

As seen above the price distribution is heavily right skewed, this usually brings bias to our models, we can easily check this with a Quantile-Quantile Plot, and when non-normality is found, we can check if the log of the data becomes normal:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2010.png)

On the top left we can see that the data is non-normal, as it deviates heavily from the red line, on the top right we can see the log of the price is very normal, we can use this to get a better prediction in our models.

The same analysis has been carried out in all numerical features.

### Empty values

Sometimes the data is not complete, in the Garages feature there were a few empty values, 388 to be exact, as to be able to use this feature in our models some value must be given. Just using the average would skew the data, since there are big apartments would be left with too few garages, and small ones would get more, so a ratio of garages per m² was found, and inputted into the training data.

# Which features to use?

Using too many features could clog up the models, making it biased, having too few would create this problem as well, to address this there was a few methods used.

### Correlation

The more diversification a variable has with another and the target feature (Price), the better it can train the models. For this we can easily use a correlation matrix:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2011.png)

With the exception of condominium fee - which had many empty values to begin with - all the numerical features seem to have a good linear relation with the target feature, and can be used to describe it. Furthermore, none of the features seem to completely explain one another, if this was the case we’d have to remove one of them as not to bias the models.

The difference from Spearman’s to Pearson’s correlation indicates that the Condominium Fee does not follow a linear relation to Price, analysing a scatterplot from price to condominium we see that they have a second degree relationship, this should then be adapted to the feature in case it is used in the models.

### Categorical feature selection

To fine tune the categorical feature selection a statistical analysis was conducted, it consisted in verifying if the population with and without the specific feature had very different characteristics, this analysis in specific verified means and standard deviation.

This was done with A/B testing using various methods (Shapiro to check for normality, Levene for homogeneity, T-Test for normal distributions and Mann-Whitney U for non-normal distributions), which resulted in p-values that would define if the groups with and without these features were too similar, a high p-value  (more than 0.05 or 95% confidence) indicates that the groups are similar and shouldn’t be used, while a low p-value was used to define that the feature would be used.

Example of features that were considered too similar and weren’t used in the models:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2012.png)

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2013.png)

Example of features that were deemed good for the models:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2014.png)

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2015.png)

# Training the models

### Linear regression

As a baseline for model selection we’ll use a linear regression, this fits a simple straight line for each feature. We’ll try the simple regressor which has no extra parameters, as well as lasso and ridge regressor, both of which have regularization parameters (alpha) that reduce over fitting, a few values for this parameters were used.

[Code for the linear regressor](https://www.notion.so/Code-for-the-linear-regressor-b7bf0de56497461c92d0e11767010bcb?pvs=21)

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2016.png)

In the picture above we see that the best linear regressor found was Lasso Regressor, with an mean squared error of 0.75 for both the training and test set. The top two graphs shows that predicted value (x-axis) plotted against the real value (y-axis), a perfect prediction would see all values falling on the red line.

On the bottom graphs we can see the residuals (how much the prediction differ from the real value), if there were a pattern in these residuals, it would mean there was a relation in the prices that wasn’t found, but there is no pattern to be seen.

# Random Forest Regressor

A better regressor is a random forest, it utilizes many forests, each with different features and leaf nodes, with different conditions to reach this leaf node, all these leaf nodes then “vote” on the predicted value.

<aside>
💡 Click below to check the code!

[Code for random forest regressor](https://www.notion.so/Code-for-random-forest-regressor-3f1acc77319640cb9d27a2d4b18b0b4e?pvs=21)

</aside>

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2017.png)

tWith the default values we can see that the random forest regressor already yields a much better result in predicting the true value, the residuals continue to have non-discernable pattern and the data comes a lot closer to a straight line.

Hyper parametrization was conducted and this resulted in a slightly improved score of 0.838 in the test data set.

<aside>
💡 Click below to check the Hyper parametrization code!

[Hyper parametrization code RFG](https://www.notion.so/Hyper-parametrization-code-RFG-7c6d8b8588e44a578d3fdd1b568b646b?pvs=21)

</aside>

# Other tested models

Models that were also tested but performed worse were LightGBM, XGBoost and GradientBoostRegressor.

# Chosen model: CATBoost

CATBoost had the best score with default parameters, so a more detailed hyper parametrization was conducted.

The defaults values almost all led to a better model, with exception to learning rate which managed to slight improve the R2 score.

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2018.png)

## Further residual analysis

When visualizing the residuals, it became clear that the bigger residuals where in the entries with the bigger areas, to see if there is a pattern in the predictions, another plot of the residuals against the features was conducted:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2019.png)

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2020.png)

As the tables showed, the larger the area the more inaccurate our models are, if a greater precision is needed, a new model with smaller areas could be trained.

# Finding the best deals

Back to the original problem, we wanted to find the best bang for the buck in the R$800K to R$1.2M range. For this we’ll search for apartments with a predicted value below the real value, and analyse them manually.

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2021.png)

The entries below the dashed line, 19 in total, are our objects of interest, if the model gave a good prediction, they should be undervalued.

Looking at the biggest residual of them all we see something interesting, this apartment is beach front, huge area, 4 bathrooms, with an undervaluation of R$3.3M, a great deal right? Well, it’s a scam.

The next 3 points and most of these “best deals” shows another problem with the models, they are really old apartments, something the collected data wasn’t able to show, further proving that a model is only as good as the data it receives.

Another problem found when viewing these entries is the districts feature, “Centro” includes both beach front apartments as well as “highway front” as seen below, a better predictor would be distance to the beach.

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2022.png)

Also while reviewing the options some pictures seemed familiar, 3 of the listings were using the same pictures for apartments in different parts of the city, with different areas and bedrooms, also indicating that these could be a scam.

Here’s a list of all potential places sorted by residual value:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2023.png)

# Data leakage test

To be sure that no data was leaked during the first data analysis, 3000 new rows of data were acquired and run in the previous models, this is the result:

![Untitled](Finding%20the%20right%20place%20c70df3a46dad45b5ab696513e22722f2/Untitled%2024.png)

All our models dropped in score but Catboost, this shows how robust Catboost is and that our analysis and modelling was consistent, but could be improved with more data, for example, latitude and longitude, year of construction, floor, quality of materials, number of suites, etc.