
# coding: utf-8

# # Power Plant Machine Learning Pipeline Application
# This notebook is an end-to-end exercise of performing Extract-Transform-Load and
# Exploratory Data Analysis on a real-world dataset, and then applying several
# different machine learning algorithms to solve a supervised regression problem
# on the dataset.
# 
# **This notebook covers:**
# 
# * *Part 1: Business Understanding*
# * *Part 2: Load Your Data*
# * *Part 3: Explore Your Data*
# * *Part 4: Visualize Your Data*
# * *Part 5: Data Preparation*
# 
# 
# *Our goal is to accurately predict power output given a set of environmental
# readings from various sensors in a natural gas-fired power generation plant.*
# 
# 
# **Background**
# 
# Power generation is a complex process, and understanding and predicting power
# output is an important element in managing a plant and its connection to the
# power grid. The operators of a regional power grid create predictions of power
# demand based on historical information and environmental factors (e.g.,
# temperature). They then compare the predictions against available resources
# (e.g., coal, natural gas, nuclear, solar, wind, hydro power plants). Power
# generation technologies such as solar and wind are highly dependent on
# environmental conditions, and all generation technologies are subject to planned
# and unplanned maintenance.
# 
# Here is an real-world example of predicted demand (on two time scales), actual
# demand, and available resources from the California power grid:
# <http://www.caiso.com/Pages/TodaysOutlook.aspx>
# 
# ![](http://content.caiso.com/outlook/SP/ems_small.gif)
# 
# The challenge for a power grid operator is how to handle a shortfall in
# available resources versus actual demand. There are three solutions to  a power
# shortfall: build more base load power plants (this process can take many years
# to decades of planning and construction), buy and import power from other
# regional power grids (this choice can be very expensive and is limited by the
# power transmission interconnects between grids and the excess power available
# from other grids), or turn on small [Peaker or Peaking Power
# Plants](https://en.wikipedia.org/wiki/Peaking_power_plant). Because grid
# operators need to respond quickly to a power shortfall to avoid a power outage,
# grid operators rely on a combination of the last two choices. In this exercise, we'll focus on the last choice.
# 
# **The Business Problem**
# 
# Because they supply power only occasionally, the power supplied by a peaker
# power plant commands a much higher price per kilowatt hour than power from a
# power grid's base power plants. A peaker plant may operate many hours a day, or
# it may operate only a few hours per year, depending on the condition of the
# region's electrical grid. Because of the cost of building an efficient power
# plant, if a peaker plant is only going to be run for a short or highly variable
# time it does not make economic sense to make it as efficient as a base load
# power plant. In addition, the equipment and fuels used in base load plants are
# often unsuitable for use in peaker plants because the fluctuating conditions
# would severely strain the equipment.
# 
# The power output of a peaker power plant varies depending on environmental
# conditions, so the business problem is _predicting the power output of a peaker
# power plant as a function of the environmental conditions_ -- since this would
# enable the grid operator to make economic tradeoffs about the number of peaker
# plants to turn on (or whether to buy expensive power from another grid).
# 
# Given this business problem, we need to first perform Exploratory Data Analysis
# to understand the data and then translate the business problem (predicting power
# output as a function of environmental conditions) into a Machine Learning task.
# In this instance, the ML task is regression since the label (or target) we are
# trying to predict is numeric. We will use an [Apache Spark ML
# Pipeline](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark-ml-package) to perform the regression.
# 
# The real-world data we are using in this notebook consists of 9,568 data points,
# each with 4 environmental attributes collected from a Combined Cycle Power Plant
# over 6 years (2006-2011), and is provided by the University of California,
# Irvine at [UCI Machine Learning Repository Combined Cycle Power Plant Data
# Set](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant). You
# can find more details about the dataset on the UCI page, including the following
# background publications:
# 
# * Pinar Tüfekci, [Prediction of full load electrical power output of a base load
# operated combined cycle power plant using machine learning
# methods](http://www.journals.elsevier.com/international-journal-of-electrical-
# power-and-energy-systems/), International Journal of Electrical Power & Energy
# Systems, Volume 60, September 2014, Pages 126-140, ISSN 0142-0615.
# * Heysem Kaya, Pinar Tüfekci and Fikret S. Gürgen: [Local and Global Learning
# Methods for Predicting Power of a Combined Gas & Steam
# Turbine](http://www.cmpe.boun.edu.tr/~kaya/kaya2012gasturbine.pdf), Proceedings
# of the International Conference on Emerging Trends in Computer and Electronics
# Engineering ICETCEE 2012, pp. 13-18 (Mar. 2012, Dubai).
# 
# **Note**:  
# For more in-depth details always refer to the documentation for [Spark Machine Learning
# Pipeline](https://spark.apache.org/docs/latest/ml-pipeline.html).
# 
# Initialize the Spark environment with the following code:

# In[1]:


import pixiedust


# To enable monitoring of Spark via the notebook:

# In[2]:


pixiedust.enableJobMonitor()


# Now there is a Spark Session, named `spark` that available for this notebook. Check it out  here:

# In[3]:


spark


# **Question**: In which mode does Spark work?
# 
# ## Part 1: Business Understanding
# The first step in any machine learning task is to understand the business need.
# 
# As described in the overview we are trying to predict power output given a set
# of readings from various sensors in a gas-fired power generation plant.
# 
# The dataset contains the following hourly average ambient variables:
# 
# - Temperature (`T`) in °C,
# - Ambient Pressure (`AP`) in  milibar,
# - Relative Humidity (`RH`) as a percentage
# - Exhaust Vacuum (`V`) pressure in cm Hg
# - Net hourly electrical energy output (`EP`) in MW (megawatts)
# 
# The averages are taken from various sensors located around the plant that record the ambient variables every second. The variables are given without normalization.
# 
# 
# The problem is a regression problem since the label (or target) we are trying to predict is numeric, i.e. we want to produce a model that given as inputs `T`, `AP`, `RH` and `V` will produce estimates of `EP`.
# 
# $ EP = f(T, AP, RH, V) $ 
# 
# A graphical representation of the model is below.
# 
# ![A graphical representation of the model](images/model.001.jpeg)
# 
# ## Part 2: Extract-Transform-Load (ETL) Your Data
# 
# Now that we understand what we are trying to do, the first step is to load our
# data into a format we can query and use.  This is known as ETL or "Extract-
# Transform-Load".  
# 
# Our data is available in the folder `../data/powerplant`, stored in 5 files named Sheet1.csv to Sheet 5.csv
# 
# ### Exercise 1 (a)
# 
# **To Do:** Let's start by having a look on the data folder.
# 
# Browse to the [folder data/powerplant](../data/powerplant) and explore the files available.
# 
# What did we need to observe?
# 
# From our initial exploration of the data, we can make several
# observations for the ETL process:
# 
#   - The data is a set of .csv (Comma Seperated Values) files (i.e., each row of the data is separated using tabs)
#   - There is a header row, which is the name of the columns
#   - It looks like the type of the data in each column is consistent (i.e., each
# column is of type double)
# 
# 
# 
# Now, let's use Spark to load the dataset into a single dataframe.

# In[4]:


powerPlantDF = spark.read.csv('../data/powerplant/', header=True, inferSchema = True)


# **Note** that instead of pointing to a single file we point to folder! All data from all files are read together in a single dataframe!
# 
# 
# ### Exercise 2 (b)
# 
# Start exploring your dataframe, and verify the names and types of the data columns 
# 
# Check the names and types of the columns using the `dtypes` method.

# In[5]:


print (powerPlantDF.dtypes)


# Alternatively, you can examine the data using the `printSchema()` method.

# In[6]:


powerPlantDF.printSchema()


# ## Part 3: Explore Your Data
# 
# Let's get some basic statistical summary of all the columns.

# In[7]:


powerPlantDF.describe().show()


# **Question** What do you observe from the summury statistics above? Discuss in pairs. 
# 
# 
# ## Part 4: Visualize Your Data
# 
# To understand our data, we will look for correlations between inputs and the
# output.  
# This can be important when choosing a model.  E.g., if features and a
# label are linearly correlated, a linear model like Linear Regression can do
# well; if the relationship is very non-linear, more complex models such as
# Decision Trees can be better. 
# 
# 
# For the visualization, we will use Pixiedust, and the embedded `display`  function.
# 
# 
# ### Exercise 4(a)
# 
# 
# Let's start with examining if there is a correlation between Temperature and Power Output. 
# 
# In the cell below, visualize the `powerPlantDF` dataframe. 
# What we will see first, is a tabular view of the `powerPlantDF`. Note that not the whole dataset is visualized, rather 100 records only.

# In[8]:


display(powerPlantDF)


# Then, lets create a scatter plot to visualize the relationships between temperature and power output.
# 
# First, select the type of the diagram, from the dropdown menu, as below:
# 
# ![Step 1](images/step1.jpeg)
# 
# Then, follow the scatter plot option wizard, below. Drag and drop the columns Temperature (`AT`) and Power Output (`PE`) as keys and values to display. Select also how big your sample should be - 500 rows should be fine for now! 
# 
# 
# ![Step 2](images/step2.jpeg)
# 
# 
# 
# 
# It looks like there is strong linear correlation between Temperature and Power
# Output.
# 
# **ASIDE: A quick physics lesson**: This correlation is to be expected as the
# second law of thermodynamics puts a fundamental limit on the [thermal
# efficiency](https://en.wikipedia.org/wiki/Thermal_efficiency) of all heat-based
# engines. The limiting factors are:
# 
#  - The temperature at which the heat enters the engine ($T_{H}$)
#  - The temperature of the environment into which the engine exhausts its waste
# heat ( $T_C$)
# 
# Our temperature measurements are the temperature of the environment. From
# [Carnot's
# theorem](https://en.wikipedia.org/wiki/Carnot%27s_theorem_%28thermodynamics%29),
# no heat engine working between these two temperatures can exceed the Carnot
# Cycle efficiency:
# $n_{th} \le 1 - \frac{T_C}{T_H}$
# 
# Note that as the environmental temperature increases, the efficiency decreases
# -- _this is the effect that we see in the above graph._
# 
# ### Exercise 4(b)
# 
# Continue exploring the relationships (if any) between the variables and
# Power Output.
# For example, you could plot of Power(PE) as a function of Exhaust Vacuum
# (V), Pressure (AP) and  Humidity (RH).

# In[9]:


<Your code here>


# ## Part 5: Data Preparation
# 
# The next step is to prepare the data for machine learning. Since all of this
# data is numeric and consistent this is a simple and straightforward task. In your group work this wont always be the case, so you need to clean up the data first!
# 
# The goal is to use machine learning to determine a function that yields the
# output power as a function of a set of predictor features. Recall the ML process, below.
# 
# ![](images/Pipeline.jpg)
# 
# The first step in building our ML pipeline is to convert the input data into a set of *features*. We will do this in Spark, using the   `VectorAssembler` for 
# tranforming a set of DataFrame
# columns into a vector of features.
# 
# 
# The VectorAssembler is a transformer that combines a given list of columns into a single vector column. 
# It is useful for combining raw features and features
# generated by different feature transformers into a single feature vector.
# VectorAssembler takes a list of input column names (each is a string) and the
# name of the output column (as a string).
# 
# ### Exercise 5(a)
# 
# - Read the Spark documentation and usage examples for
# [VectorAssembler](https://spark.apache.org/docs/2.0.0/ml-features.html#vectorassembler), and convert the `powerPlantDF` to a DataFrame
# named `dataset`, so that:
# - Set the vectorizer's input columns to a list of the four columns of the input
# DataFrame: `["AT", "V", "AP", "RH"]`
# - Set the vectorizer's output column name to `"features"`
# 
# Yout code should look like:
# <pre>
# VectorAssembler(
#     inputCols=["AT", "V", "AP", "RH"],
#     outputCol="features")
# </pre>

# In[10]:


# TODO: Replace <FILL_IN> with the appropriate code
from pyspark.ml.feature import VectorAssembler

vectorizer = VectorAssembler(
    inputCols = ['AT', 'V', 'AP', 'RH'],
    outputCol = 'features')
dataset = vectorizer.transform(powerPlantDF)


# The vector assembler above, is not invasive. It transformed the powerPlantDF dataframe by appending a new column named features. Can you confirm this below, by observing the schema?

# In[13]:


powerPlantDF.printSchema()
dataset.printSchema()


# ##Part 6: Data Preparation
# Now let's model our data to predict what the power output will be given a set of
# sensor readings
# 
# Our first model will be based on simple linear regression since we saw some
# linear patterns in our data based on the scatter plots during the exploration
# stage.
# 
# We need a way of evaluating how well our linear regression model predicts power
# output as a function of input parameters. We can do this by splitting up our
# initial data set into a _Training Set_ used to train our model and a _Test Set_
# used to evaluate the model's performance in giving predictions. We can use a
# DataFrame's [randomSplit()](https://spark.apache.org/docs/2.0.0/api/python/pyspa
# rk.sql.html#pyspark.sql.DataFrame.randomSplit) method to split our dataset. The
# method takes a list of weights and an optional random seed. The seed is used to
# initialize the random number generator used by the splitting function.
# 
# ### Exercise 6(a)
# 
# Use the [randomSplit()](https://spark.apache.org/docs/1.6.2/api/python/pyspark.s
# ql.html#pyspark.sql.DataFrame.randomSplit) method to divide up `datasetDF` into
# a trainingSetDF (80% of the input DataFrame) and a testSetDF (20% of the input
# DataFrame), and for reproducibility, use the seed 1800009193. Then cache each
# DataFrame in memory to maximize performance.

# In[23]:


# TODO: Replace <FILL_IN> with the appropriate code.
# We'll hold out 20% of our data for testing and leave 80% for training
seed = 1800009193
(split20DF, split80DF) = dataset.randomSplit([.2, .8], seed)

# Let's cache these datasets for performance
testSetDF = split20DF
trainingSetDF = split80DF


# ## Part 7: Linear Regression Model
# 
# From the Wikipedia article on [Linear
# Regression](https://en.wikipedia.org/wiki/Linear_regression):
# > In statistics, linear regression is an approach for modeling the relationship
# between a scalar dependent variable \\( y \\) and one or more explanatory
# variables (or independent variables) denoted \\(X\\). In linear regression, the
# relationships are modeled using linear predictor functions whose unknown model
# parameters are estimated from the data. Such models are called linear models.
# 
# Linear regression has many practical uses. Most applications fall into one of
# the following two broad categories:
#   - If the goal is prediction, or forecasting, or error reduction, linear
# regression can be used to fit a predictive model to an observed data set of
# \\(y\\) and \\(X\\) values. After developing such a model, if an additional
# value of \\(X\\) is then given without its accompanying value of \\(y\\), the
# fitted model can be used to make a prediction of the value of \\(y\\).
#   - Given a variable \\(y\\) and a number of variables \\( X_1 \\), ..., \\( X_p
# \\) that may be related to \\(y\\), linear regression analysis can be applied to
# quantify the strength of the relationship between \\(y\\) and the \\( X_j\\), to
# assess which \\( X_j \\) may have no relationship with \\(y\\) at all, and to
# identify which subsets of the \\( X_j \\) contain redundant information about
# \\(y\\).
# 
# We are interested in both uses, as we would like to predict power output as a
# function of the input variables, and we would like to know which input variables
# are weakly or strongly correlated with power output.
# 
# Since Linear Regression is simply a Line of best fit over the data that
# minimizes the square of the error, given multiple input dimensions we can
# express each predictor as a line function of the form:
# 
# \\[ y = a + b x_1 + b x_2 + b x_i ... \\]
# 
# where \\(a\\) is the intercept and the \\(b\\) are the coefficients.
# 
# 
# In Spark, a Linear regression model is another step in a ML pipeline. 
# 
# Run the cells below to create a Linear Regression Model

# In[16]:


from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml import Pipeline

 # Initialize the linear regression learner with default values for the parameters
lr = LinearRegression()


# Each learning algorithm, as the Linear Regression Model, has some parameters
# that affect how learning is done. This is custom for each learner.
# 
# In the case of Linear Regression we have:

# In[17]:


# See which are the parameters
print(lr.explainParams())


# Two parameters are not optional:
# - The name of the label column to "PE" (i.e. which are the known values to
# learn)
# - The name of the prediction column to "Predicted_PE" (i.e. where the
# predictions values should be stored)

# In[18]:


lr.setPredictionCol("Predicted_PE")  .setLabelCol("PE")


# We will also configure two parameters, which a re customary to the linear
# regression
# - the maximum number of iterations to 100
# - the regularization parameter to 0.1

# In[19]:


lr.setMaxIter(100)  .setRegParam(0.1)


# ## Part 8 Create a pipeline
# 
# Next, to create a workflow that puts together the vectorization and the Linear
# Regression learner, we can create an ML Pipeline that stitch together the two
# trasnformations we created before

# In[25]:


lrPipeline = Pipeline()
lrPipeline.setStages([vectorizer, lr])


# ### Exercise 8 (a) Train with the training dataset the model
# 
# Next, we create a Linear Regression model that has been trained (or *fit*) with
# the training data set.
# To do so we apply the `lrPipeline` pipeline of the training dataset, i.e. first
# vectorize and then train with the linear regression model:

# In[28]:


# Let's first train on the trqining dataset to see what we get
lrModel = lrPipeline.fit(trainingSetDF)


# ### Exercise 8 (b) Inspect the model
# The learner has been trained now. Lets inspect which are the wrights for the
# (trained) linear regression model, which is now stored as the second element of
# our pipeline.
# 
# Run the next cell. Ensure that you understand what's going on. Ask for help if
# you have questions.

# In[ ]:


# The coefficients (i.e., weights) are as follows:
weights = lrModel.stages[1].coefficients

# The corresponding features for these weights are:
featuresNoLabel = vectorizer.getInputCols()


# Print coefficients 
list(zip(featuresNoLabel, weights))
 
 # Print the intercept
print(lrModel.stages[1].intercept)


# **Exercises**
# 
# - Write down the linear regression equation that your model learned.
# - Recall when we visualized each predictor against Power Output using a Scatter
# Plot,
# does the final equation seems logical given
# those visualizations?
# 
# ## Part 9: Learner evaluation
# 
# ### Exercise 9(a) Apply the learner to make predictions
# 
# Now let's see what our predictions look like given this model. We apply our
# Linear Regression model to the 20% of the data that we split from the input
# dataset. The output of the model will be a predicted Power Output column named
# "Predicted_PE".
# 
# - Run the next cell
# - Scroll through the resulting table and notice how the values in the Power
# Output (PE) column compare to the corresponding values in the predicted Power
# Output (Predicted_PE) column

# In[ ]:


# Apply our LR model to the test data and predict power output
predictionsLR = lrModel.transform(testSetDF).select("AT", "V", "AP", "RH", "PE", "Predicted_PE")

# Print the first 15 rows of your predictions
predictionsLR.show(15) 


# From a visual inspection of the predictions, we can see that they are close to
# the actual values.
# 
# 
# However, we would like a scientific measure of how well the Linear Regression
# model is performing in accurately predicting values. To perform this
# measurement, we can use an evaluation metric such as [Root Mean Squared
# Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE) to
# validate our Linear Regression model.
# 
# RSME is defined as follows: \\( RMSE = \sqrt{\frac{\sum_{i = 1}^{n} (x_i -
# y_i)^2}{n}}\\) where \\(y_i\\) is the observed value and \\(x_i\\) is the
# predicted value
# 
# RMSE is a frequently used measure of the differences between values predicted by
# a model or an estimator and the values actually observed. The lower the RMSE,
# the better our model.
# 
# Spark ML Pipeline provides several regression analysis metrics, including [RegressionEvaluator](https://spark.apache.org/docs/2.0.0/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator).
# 
# After we create an instance of [RegressionEvaluator](https://spark.apache.org/do
# cs/2.0.0/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator),
# we set the label column name to "PE" and set the prediction column name to
# "Predicted_PE". We then invoke the evaluator on the predictions.
# 
# ### Exercise 9 (b) Model evaluation with RSME

# In[ ]:


# Now let's compute an evaluation metric for our test dataset
from pyspark.ml.evaluation import RegressionEvaluator

# Create an RMSE evaluator using the label and predicted columns
regEval = RegressionEvaluator(predictionCol="Predicted_PE", labelCol="PE", metricName="rmse")

# Run the evaluator on the DataFrame
rmse = regEval.evaluate(predictionsLR)

print("Root Mean Squared Error: %.2f" % rmse)


# ### Exercise 9(c) Model evaluation with r^2
# 
# Another useful statistical evaluation metric is the coefficient of
# determination, denoted \\(R^2\\) or \\(r^2\\) and pronounced "R squared". It is
# a number that indicates the proportion of the variance in the dependent variable
# that is predictable from the independent variable and it provides a measure of
# how well observed outcomes are replicated by the model, based on the proportion
# of total variation of outcomes explained by the model. The coefficient of
# determination ranges from 0 to 1 (closer to 1), and the higher the value, the
# better our model.
# 
# To compute \\(r^2\\), we invoke the evaluator with  `regEval.metricName: "r2"`
# 
# 
# Run the next cell and ensure that you understand what's going on.

# In[ ]:


# Now let's compute another evaluation metric for our test dataset
r2 = regEval.evaluate(predictionsLR, {regEval.metricName: "r2"})

print("r2: {0:.2f}".format(r2))


# ## Part 10: Parameter Tuning and Evaluation
# 
# Now that we have a model with all of the data let's try to make a better model
# by tuning over several parameters. The process of tuning a model is known as
# [Model Selection](https://spark.apache.org/docs/2.0.0/api/python/pyspark.ml.html#module-pyspark.ml.tuning) or [Hyperparameter
# Tuning](https://spark.apache.org/docs/2.0.0/api/python/pyspark.ml.html#module-
# pyspark.ml.tuning), and Spark ML Pipeline makes the tuning process very simple
# and easy.
# 
# An important task in ML is model selection, or using data to find the best model
# or parameters for a given task. This is also called tuning. Tuning may be done
# for individual Estimators such as
# [LinearRegression](https://spark.apache.org/docs/1.6.2/ml-classification-
# regression.html#linear-regression), or for entire Pipelines which include
# multiple algorithms, featurization, and other steps. Users can tune an entire
# Pipeline at once, rather than tuning each element in the Pipeline separately.
# 
# Spark ML Pipeline supports model selection using tools such as [CrossValidator](
# https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning
# .CrossValidator), which requires the following items:
# 
#   - [Estimator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#p
# yspark.ml.Estimator): algorithm or Pipeline to tune
#   - [Set of ParamMaps](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml
# .html#pyspark.ml.tuning.ParamGridBuilder): parameters to choose from, sometimes
# called a _parameter grid_ to search over
#   - [Evaluator](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#p
# yspark.ml.evaluation.Evaluator): metric to measure how well a fitted Model does
# on held-out test data
# 
# At a high level, model selection tools such as CrossValidator
# work as follows:
# 
#   - They split the input data into separate training and test datasets.
#   - For each (training, test) pair, they iterate through the set of ParamMaps:
#     - For each ParamMap, they fit the Estimator
# using those parameters, get the fitted Model, and evaluate the Model's
# performance using the Evaluator.
#   - They select the Model produced by the best-performing set of parameters.
# 
# The Evaluator can be a [RegressionEvaluator](https://spark.apache.org/docs/1.6.2
# /api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator) for
# regression problems. To help construct the parameter grid, users
# can use the [ParamGridBuilder](https://spark.apache.org/docs/1.6.2/api/python/py
# spark.ml.html#pyspark.ml.tuning.ParamGridBuilder) utility.
# 
# Note that cross-validation over a grid of parameters is expensive. For example,
# in the next cell, the parameter grid has 10 values for
# [lr.regParam](https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pys
# park.ml.regression.LinearRegression.regParam), and CrossValidator uses 3 folds.
# This
# multiplies out to (10 x 3) = 30 different models being trained. In realistic
# settings, it can be common to try many more parameters (e.g., multiple values
# for multiple parameters) and use more folds (_k_ = 3 and _k_ = 10 are common).
# In other words, using CrossValidator can be very expensive.
# However, it is also a well-established method for choosing parameters which is
# more statistically sound than heuristic hand-tuning.
# 
# 
# ### Exercise 10 (a)
# We perform the following steps:
# 
#   - Create a CrossValidator using the Pipeline and RegressionEvaluator that we
# created earlier, and set the
# number of folds to 3
#   - Create a list of 10 regularization parameters
#   - Use ParamGridBuilder to build a parameter grid with the
# regularization parameters and add the grid to the CrossValidator
#   - Run the CrossValidator to find the parameters that yield
# the best model (i.e., lowest RMSE) and return the best model.
# 
# 
# Run the next cell. _Note that it will take some time to run the CrossValidator
# as it will run almost 200 Spark jobs!_
# 
# Expand the Spark Job Monitor, to visualize what is going on in the background!

# In[ ]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

 # We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
 # Let's create our CrossValidator with 3 fold cross validation
crossval = CrossValidator(estimator=lrPipeline, evaluator=regEval, numFolds=3)

 # Let's tune over our regularization parameter from 0.01 to 0.10
regParam = [x / 100.0 for x in range(1, 11)]

 # We'll create a paramter grid using the ParamGridBuilder, and add the grid to the CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, regParam)
             .build())
crossval.setEstimatorParamMaps(paramGrid)

 # Now let's find and return the best model
cvModel = crossval.fit(trainingSetDF).bestModel


# ### Exercise 10 (b)
# 
# Now that we have tuned our Linear Regression model, let's see what the new RMSE
# and \\(r^2\\) values are for these models, and compare with our initial model.
# 
# 
# Complete and run the next cell.

# In[ ]:


# TODO: Replace <FILL_IN> with the appropriate code.
# Now let's use cvModel to compute an evaluation metric for our test dataset: testSetDF
predictionsRL = <FILL_IN>

# Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
rmseLR = <FILL_IN>

# Now let's compute the r2 evaluation metric for our test dataset
r2LR = <FILL_IN>

print("Original Root Mean Squared Error: {0:2.2f}".format(rmse))
print("New Root Mean Squared Error: {0:2.2f}".format(rmseLR))
print("Old r2: {0:2.2f}".format(r2))
print("New r2: {0:2.2f}".format(r2LR))


# **Discussion**  
# How does the initially untuned model compare with the tuned model?
# Are they statistically similar?
