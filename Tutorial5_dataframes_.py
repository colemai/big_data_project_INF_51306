
# coding: utf-8

# # Spark Dataframes
# 
# RDDs are the building blocks of Spark. It’s the original API that Spark exposed
# and pretty much all the higher level APIs decompose to RDDs. The advantages of 
# RDDs are manifold, but there are also some problems. For example, it’s easy to 
# build inefficient transformation chains, they are slow with non-JVM languages 
# such as Python, they can not be optimized by Spark. Lastly, it’s difficult to 
# understand what is going on when you’re working with them, because, for example, 
# the transformation chains are not very readable in the sense that you don’t 
# immediately see what will be the solution, but how you are doing it.
# 
# #### DataFrame
# 
# Because of the disadvantages that you can experience while working with RDDs, 
# the DataFrame API was conceived: it provides you with a higher level abstraction 
# that allows you to use a query language to manipulate the data. This higher level 
# abstraction is a logical plan that represents data and a schema. This means that 
# the frontend to interacting with your data is a lot easier! Because the logical 
# plan will be converted to a physical plan for execution, you’re actually a lot 
# closer to what you’re doing when you’re working with them rather than how you’re 
# trying to do it, because you let Spark figure out the most efficient way to do 
# what you want to do.
# 
# Remember though that DataFrames are still built on top of RDDs!
# 
# While you still can interact directly with RDDs, **DataFrames are preferred**. They're
# generally faster, and they perform the same no matter what language (Python, R,
# Scala or Java) you use with Spark.
# 
# In this set of tutorials, we'll learn how to use DataFrames, and the following
# transformations will be covered:
# 
# - `select()`, `filter()`, `distinct()`, `dropDuplicates()`, `orderBy()`,
# `groupBy()`
# 
# The following actions will be covered:
# - `first()`, `take()`, `count()`, `collect()`, `show()`
# 
# Also covered:
# - `cache()`, `unpersist()`
# 
# 
# ## Part 1: Using DataFrames and chaining together transformations and actions
# 
# ### Working with your first DataFrames
# 
# The entry point for using data frames is the [SQLContext](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SQLContext)

# In[1]:


import pixiedust
from pixiedust import sc
pixiedust.enableJobMonitor()

sqlContext = pixiedust.SQLContext(sc)


# In Spark, we first create a base [DataFrame](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame). 
# We can then apply one ormore transformations to that base DataFrame. 
# *A DataFrame is, just like RDDs, immutable, so once it is created, it cannot be changed.* 
# As a result, each transformation creates a new DataFrame. 
# Finally, we can apply one or more actions to the DataFrames.
# 
# > Note again that Spark uses lazy evaluation, so transformations are not actually executed until an action occurs.
# 
# We will perform several exercises to obtain a better understanding of
# DataFrames:
# 
# * Read a file containing 10,000 personal records
# * Create a Spark DataFrame from that collection
# * Subtract one from each value using `map`
# * Perform action `collect` to view results
# * Perform action `count` to view counts
# * Apply transformation `filter` and view results with `collect`
# * Learn about lambda functions
# * Explore how lazy evaluation works and the debugging challenges that it
# introduces
# 
# A DataFrame consists of a series of `Row` objects; each `Row` object has a set
# of named columns. 
# You can think of a DataFrame as modeling a table, though the data source being processed does not have to be a table.
# 
# More formally, a DataFrame must have a _schema_, which means it must consist of
# columns, each of which has a _name_ and a _type_. 
# Some data sources have schemas built into them. 
# Examples include RDBMS databases, Parquet files, and NoSQL
# databases like Cassandra. 
# Other data sources don't have computer-readable schemas, but you can often apply a schema programmatically.
# 
# # Long Example
# ## A dataset of 10.000 random people
# 
# A collection has been created consisting of random data of fake person records.
# This collection is available in the data folder.
# 
# First, we will create a list containing tuples with this data.

# In[2]:


# Open persons.txt
data = []
import csv

# Open persons.txt
with open('../data/persons.txt') as person_file:
    # Iterate through the file
    for line in csv.reader(person_file, delimiter=';'):
        # Convert last value (age) into int
        line[-1] = int(line[-1])
        # Append values to data list
        data.append(tuple(line))


# `data` is just a normal Python list, containing Python tuples objects. Let's
# look at the first item in the list:

# In[3]:


data[0]


# We can check the size of the list using the Python `len()` function.

# In[4]:


len(data)


# ## Create a DataFrame from a collection
# 
# In Spark, datasets are represented as a list of entries, where the list is
# broken up into many different partitions that are each stored on a different
# machine.  Each partition holds a unique subset of the entries in the list.
# 
# One of the defining features of Spark, compared to other data analytics
# frameworks (e.g., Hadoop), is that it stores data in memory rather than on disk.
# This allows Spark applications to run much more quickly, because they are not
# slowed down by needing to read data from disk.
# The figure below illustrates how Spark breaks a list of data entries into
# partitions that are each stored in memory on a worker (executor on the right hand side of the diagram).
# 
# ![](http://spark-mooc.github.io/web-assets/images/cs105x/diagram-3b.png)
# 
# To create the DataFrame, we'll use `sqlContext.createDataFrame()`, and we'll
# pass our array of data in as an argument to that function. Spark will create a
# new set of input data based on data that is passed in.  A DataFrame requires a
# _schema_, which is a list of columns, where each column has a name and a type.
# Our list of data has elements with types (mostly strings, but one integer).
# 
# We'll supply the rest of the schema and the column names as the second argument
# to `createDataFrame()`.

# In[6]:


dataDF = sqlContext.createDataFrame(data, ('last_name', 'first_name', 'ssn', 'occupation', 'age'))


# Let's see what type `sqlContext.createDataFrame()` returned.

# In[7]:


print('type of dataDF: %s', type(dataDF))


# Let's take a look at the DataFrame's schema and some of its rows.

# In[8]:


dataDF.printSchema()


# On how many partitions is this DataFrame split into?

# In[9]:


dataDF.rdd.getNumPartitions()


# ### Use _select_ to retrieve data
# 
# So far, we've created a distributed DataFrame that is split into many
# partitions, where each partition is stored on a single machine in our cluster.
# Let's look at what happens when we do a basic operation on the dataset.  Many
# useful data analysis operations can be specified as "do something to each item
# in the dataset".  These data-parallel operations are convenient because each
# item in the dataset can be processed individually: the operation on one entry
# doesn't effect the operations on any of the other entries.  Therefore, Spark can
# parallelize the operation.
# 
# One of the most common DataFrame operations is `select()`, and it works more or
# less like a SQL `SELECT` statement: You can select specific columns from the
# DataFrame, and you can even use `select()` to create _new_ columns with values
# that are derived from existing column values. We can use `select()` to create a
# new column that decrements the value of the existing `age` column.
# 
# Note that `select()` is a _transformation_. It returns a new DataFrame that captures both
# the previous DataFrame and the operation to add to the query (`select`, in this
# case). 
# But it does *not* actually execute anything on the cluster. When
# transforming DataFrames, we are building up a _query plan_. That query plan will
# be optimized, implemented (in terms of RDDs), and executed by Spark _only_ when
# we call an action.

# In[10]:


# Transform dataDF through a select transformation and rename the newly created '(age -1)' column to 'age'
# Because select is a transformation and Spark uses lazy evaluation, no jobs, stages,
# or tasks will be launched when we run this code.
subDF = dataDF.select('last_name', 'first_name', 'ssn', 'occupation', (dataDF.age - 1).alias('age'))


# This transformation is lazy! (of course!). i.e. nothing has been computed yet.
# Let's take a look at `subDF`.

# In[11]:


subDF


# ### Use _collect_ to view results
# 
# ![Collect](http://spark-mooc.github.io/web-assets/images/cs105x/diagram-3d.png)
# 
# To see a list of elements decremented by one, we need to create a new list on
# the driver from the the data distributed in the executor nodes.  To do this we
# can call the `collect()` method on our DataFrame.  `collect()` is often used
# after transformations to ensure that we are only returning a *small* amount of
# data to the driver.  This is done because the data returned to the driver must
# fit into the driver's available memory.  If not, the driver will crash.
# 
# The `collect()` method is the first action operation that we have encountered.
# Action operations cause Spark to perform the (lazy) transformation operations
# that are required to compute the values returned by the action.  In our example,
# this means that tasks will now be launched to perform the `createDataFrame`,
# `select`, and `collect` operations.
# 
# In the diagram, the dataset is broken into four partitions, so four `collect()`
# tasks are launched. Each task collects the entries in its partition and sends
# the result to the driver, which creates a list of the values, as shown in the
# figure below.
# 
# Now let's run `collect()` on `subDF`.
# 
# Check via the job scheduler, that this task is executed in 10 stages - as many as the partitions of this dataframe!

# In[12]:


# Let's collect the data
results = subDF.collect()
print(results)


# A better way to visualize the data is to use the `show()` method. If you don't
# tell `show()` how many rows to display, it displays 20 rows.

# In[13]:


subDF.show()


# ### Use _count_ to get total
# 
# One of the most basic jobs that we can run is the `count()` job which will count
# the number of elements in a DataFrame, using the `count()` action. Since
# `select()` creates a new DataFrame with the same number of elements as the
# starting DataFrame, we expect that applying `count()` to each DataFrame will
# return the same result.
# 
# 
# 
# Note that because `count()` is an action operation, if we had not already
# performed an action with `collect()`, then Spark would now perform the
# transformation operations when we executed `count()`.
# 
# The figure below, shows how it works.
# 
# ![Count](http://spark-mooc.github.io/web-assets/images/cs105x/diagram-3e.png)
# 
# Each task counts the entries in its partition and sends the result to your
# SparkContext, which adds up all of the counts. The figure above shows
# what would happen if we ran `count()` on a small example dataset with just four
# partitions.

# In[ ]:


print(dataDF.count())
print(subDF.count())


# ### Apply transformation _filter_ and view results with _collect_
# 
# Next, we'll create a new DataFrame that only contains the people whose ages are
# less than 10. To do this, we'll use the `filter()` transformation. (You can also
# use `where()`, an alias for `filter()`, if you prefer something more SQL-like).
# The `filter()` method is a transformation operation that creates a new DataFrame
# from the input DataFrame, keeping only values that match the filter expression.
# 
# The figure shows how this might work on the small four-partition dataset.
# 
# ![Filter](http://spark-mooc.github.io/web-assets/images/cs105x/diagram-3f.png)
# 
# 
# To view the filtered list of elements less than 10, we need to create a new list
# on the driver from the distributed data on the executor nodes.  We use the
# `collect()` method to return a list that contains all of the elements in this
# filtered DataFrame to the driver program.

# In[ ]:


filteredDF = subDF.filter(subDF.age < 10)
filteredDF.show(truncate=False)
print("Less than 10 years are", filteredDF.count(), "people")


# (These are some _seriously_ precocious children...)
# 
# # Python lambda functions and User Defined Functions
# 
# In a previous tutorial, you learned to use lambda functions in your Python map 
# and reduce. You can also apply those with Spark DataFrames as in:

# In[ ]:


less_ten_lambda = lambda s: s < 10
lambdaDF = subDF.filter(less_ten_lambda(subDF.age))
lambdaDF.show()
lambdaDF.count()


# Past versions of DataFrames used to wrap lambdas around
# Spark _User Defined Function_ (UDF). A UDF is a special wrapper around a
# function, allowing the function to be used in a DataFrame query,
# and requires both the function and the return type to be defined.

# In[ ]:


from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf

less_ten = udf(lambda s: s < 10, BooleanType())
lambdaDF = subDF.filter(less_ten(subDF.age))
lambdaDF.show()
lambdaDF.count()


# Lets try another example below.

# In[ ]:


# Let's collect the even values less than 10
even = udf(lambda s: s % 2 == 0, BooleanType())
evenDF = lambdaDF.filter(even(lambdaDF.age))
evenDF.show()
evenDF.count()


# **Exercise**
# 
# You can rewrite some of the examples above, using lambdas!

# In[ ]:


# Transform dataDF through a select transformation and rename the newly created '(age -1)' column to 'age'
 
 


# # Additional DataFrame actions
# 
# Let's investigate some additional actions:
# 
# - [`first()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.first)
# - [`take()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.take)
# 
# One useful thing to do when we have a new dataset is to look at the first few
# entries to obtain a rough idea of what information is available.  In Spark, we
# can do that using actions like `first()`, `take()`, and `show()`. Note that for
# the `first()` and `take()` actions, the elements that are returned depend on how
# the DataFrame is *partitioned*.
# 
# Instead of using the `collect()` action, we can use the `take(n)` action to
# return the first _n_ elements of the DataFrame. The `first()` action returns the
# first element of a DataFrame, and is equivalent to `take(1)[0]`.

# In[ ]:


print("first: ", filteredDF.first(), "\n")

print("Four of them: ", filteredDF.take(4))


# This looks better:

# In[ ]:


filteredDF.show(4)


# ## Additional DataFrame transformations
# 
# ### _orderBy_
# 
# [`orderBy()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.distinct) 
# allows you to sort a DataFrame by one or more columns, producing a new DataFrame.
# 
# For example, let's get the first five oldest people in the original (unfiltered)
# DataFrame. We can use the `orderBy()` transformation. `orderBy` takes one or
# more columns, either as _names_ (strings) or as `Column` objects. To get a
# `Column` object, we use one of two notations on the DataFrame:
# 
# - Pandas-style notation: `filteredDF.age`
# - Subscript notation: `filteredDF['age']`
# 
# Both of those syntaxes return a `Column`, which has additional methods like
# `desc()` (for sorting in descending order) or `asc()` (for sorting in ascending
# order, which is the default).
# 
# Here are some examples:

# In[ ]:


# sort by age in ascending order; 
# returns a new DataFrame
dataDF.orderBy(dataDF['age'])  

# sort by last name in descending order
dataDF.orderBy(dataDF.last_name.desc()) 


# In[ ]:


# Get the five oldest people in the list. To do that, sort by age in descending order.
dataDF.orderBy(dataDF.age.desc()).take(5)


# Or use `show` for nicer printing:

# In[ ]:


# Get the five oldest people in the list. To do that, sort by age in descending order.
dataDF.orderBy(dataDF.age.desc()).show(5)


# Note that the results may not the same! Why?
# 
# **Exercise**  
# Write down the name of the five oldest people, and then run the cell above `dataDF.orderBy(dataDF.age.desc()).show(5)` again. What do you notice? Explain why.
# 
# **Exercise**  
# 
# Count how many persons have the maximum age.

# In[ ]:


# your code here


# Let's reverse the sort order. Since ascending sort is the default, we can
# actually use a `Column` object expression or a simple string, in this case. The
# `desc()` and `asc()` methods are only defined on `Column`. Something like
# `orderBy('age'.desc())` would not work, because there's no `desc()` method on
# Python string objects. That's why we needed the column expression. But if we're
# just using the defaults, we can pass a string column name into `orderBy()`. This
# is sometimes easier to read!

# In[ ]:


dataDF.orderBy('age').show(5)


# ### _distinct_ and _dropDuplicates_
# 
# [`distinct()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#p
# yspark.sql.DataFrame.distinct) filters out duplicate rows, and it considers all
# columns. Since our data is completely randomly generated,
# it's extremely unlikely that there are any duplicate rows:

# In[ ]:


print(dataDF.count())
print(dataDF.distinct().count())


# To demonstrate `distinct()`, let's create a quick throwaway dataset.

# In[ ]:


tempDF = sqlContext.createDataFrame([("Joe", 1), ("Joe", 1), ("Anna", 15), ("Anna", 12), ("Ravi", 5)], ('name', 'score'))


# In[ ]:


tempDF.show()


# In[ ]:


tempDF.distinct().show()


# Note that one of the ("Joe", 1) rows was removed, but both rows with name "Anna"
# were kept, because all columns in a row must match another row for it to be
# considered a duplicate.
# 
# [`dropDuplicates()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.dropDuplicates) 
# is like `distinct()`, except that it
# allows us **to specify the columns to compare**. For instance, we can use it to drop
# all rows where the first name and last name duplicates (ignoring the occupation
# and age columns).

# In[ ]:


print(dataDF.count())
print (dataDF.dropDuplicates(['first_name', 'last_name']).count())


# Note that with `dropDuplicates` you are not selecting columns! The dataframe still contains all the columns of the original one. Can you verify below?

# In[ ]:


df1 = dataDF.dropDuplicates(['first_name', 'last_name'])
# print the schema and/or show some rows



# ### _drop_
# 
# [`drop()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.drop) 
# is like the opposite of `select()`: Instead of selecting
# specific columns from a DataFrame, it drops a specified column from a DataFrame.
# 
# Here's a simple use case: Suppose you're reading from a 1,000-column CSV file,
# and you have to get rid of five of the columns. Instead of selecting 995 of the
# columns, it's easier just to drop the five you don't want.

# In[ ]:


dataDF.drop('occupation').drop('age').show()


# ### _groupBy_
# 
# [`groupBy()`]((http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#p
# yspark.sql.DataFrame.groupBy) is one of the most powerful transformations. It
# allows you to perform aggregations on a DataFrame.
# 
# Unlike other DataFrame transformations, `groupBy()` does _not_ return a
# DataFrame. Instead, it returns a special [GroupedData](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData) 
# object that contains various aggregation functions.
# 
# The most commonly used aggregation function is [`count()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.count),
# but there are others (like [`sum()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.sum), 
# [`max()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.max), 
# and [avg()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData.avg).
# 
# These aggregation functions typically create a new column and return a new
# DataFrame.

# In[ ]:


dataDF.groupBy('occupation').count().show(truncate=False)


# In[ ]:


dataDF.groupBy().avg('age').show(truncate=False)


# We can also use `groupBy()` to do another useful aggregations:

# In[ ]:


print("Maximum age:", dataDF.groupBy().max('age').first()[0])
print("Minimum age:", dataDF.groupBy().min('age').first()[0])


# ### _sample_
# 
# When analyzing big data, the [`sample()`](http://spark.apache.org/docs/latest/api/py
# thon/pyspark.sql.html#pyspark.sql.DataFrame.sample) transformation is often
# quite useful. It returns a new DataFrame with a random sample of elements from
# the dataset.  It takes in a `withReplacement` argument, which specifies whether
# it is okay to randomly pick the same item multiple times from the parent
# DataFrame (so when `withReplacement=True`, you can get the same item back
# multiple times). It takes in a `fraction` parameter, which specifies the
# fraction elements in the dataset you want to return. (So a `fraction` value of
# `0.20` returns 20% of the elements in the DataFrame.) It also takes an optional
# `seed` parameter that allows you to specify a seed value for the random number
# generator, so that reproducible results can be obtained.

# In[ ]:


sampledDF = dataDF.sample(withReplacement=False, fraction=0.10)
print(sampledDF.count())
sampledDF.show()


# In[ ]:


print(dataDF.sample(withReplacement=False, fraction=0.05).count())


# **Question**  
# The result of count may seem a bit odd. Can you think why?
# 
# ### Data partitioning
# 
# Spark has a dedicated method for splitting randomly a dataset into
# partitions.

# In[ ]:


(trainData, valData, testData) = dataDF.randomSplit([0.6, 0.2, 0.2])


# Check the sizes of the three dataframes created below.

# In[ ]:


print(trainData.count())
# do the same for the rest splits



# What happens? Can you explain why?
# 
# # Caching DataFrames and storage options
# 
# ## (a) Caching DataFrames
# 
# For efficiency Spark keeps your DataFrames in memory. (More formally, it keeps
# the _RDDs_ that implement your DataFrames in memory.) By keeping the contents in
# memory, Spark can quickly access the data. However, memory is limited, so if you
# try to keep too many partitions in memory, Spark will automatically delete
# partitions from memory to make space for new ones. If you later refer to one of
# the deleted partitions, Spark will automatically recreate it for you, but that
# takes time.
# 
# So, if you plan to use a DataFrame more than once, then you should tell Spark to
# cache it. You can use the `cache()` operation to keep the DataFrame in memory.
# However, you must still trigger an action on the DataFrame, such as `collect()`
# or `count()` before the caching will occur. In other words, `cache()` is lazy:
# It merely tells Spark that the DataFrame should be cached _when the data is
# materialized_. You have to run an action to materialize the data; the DataFrame
# will be cached as a side effect. The next time you use the DataFrame, Spark will
# use the cached data, rather than recomputing the DataFrame from the original
# data.
# 
# You can see your cached DataFrame in the "Storage" section of the Spark web UI.
# If you click on the name value, you can see more information about where the the
# DataFrame is stored.

# In[ ]:


# Cache the DataFrame
filteredDF.cache()
# Trigger an action
print(filteredDF.count())
# Check if it is cached
print(filteredDF.is_cached)


# ## (b) Unpersist and storage options
# 
# Spark automatically manages the partitions cached in memory. If it has more
# partitions than available memory, by default, it will evict older partitions to
# make room for new ones. For efficiency, once you are finished using cached
# DataFrame, you can optionally tell Spark to stop caching it in memory by using
# the DataFrame's `unpersist()` method to inform Spark that you no longer need the
# cached data.
# 
# **Advanced:**  
# Spark provides many more options for managing how DataFrames
# cached. For instance, you can tell Spark to spill cached partitions to disk when
# it runs out of memory, instead of simply throwing old ones away. You can explore
# the API for DataFrame's [`persist()`](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.persist) 
# operation using Python's [`help()`](https://docs.python.org/2/library/functions.html?highlight=help#help)
# command. The `persist()` operation, optionally, takes a pySpark [StorageLevel](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.StorageLevel)
# object.

# In[ ]:


# If we are done with the DataFrame we can unpersist it so that its memory can be reclaimed
filteredDF.unpersist()
# Check if it is cached
print(filteredDF.is_cached)


# # Debugging Spark applications and lazy evaluation
# 
# ## (a) How Python is Executed in Spark
# 
# Internally, Spark executes using a Java Virtual Machine (JVM). pySpark runs
# Python code in a JVM using [Py4J](http://py4j.sourceforge.net). Py4J enables
# Python programs running in a Python interpreter to dynamically access Java
# objects in a Java Virtual Machine. Methods are called as if the Java objects
# resided in the Python interpreter and Java collections can be accessed through
# standard Python collection methods. Py4J also enables Java programs to call back
# Python objects.
# 
# Because pySpark uses Py4J, coding errors often result in a complicated,
# confusing stack trace that can be difficult to understand. In the following
# section, we'll explore how to understand stack traces.
# 
# ## (b) Challenges with lazy evaluation using transformations and actions
# 
# Spark's use of lazy evaluation can make debugging more difficult because code is
# not always executed immediately. 
# To see an example of how this can happen, let's first define a broken filter function.
# Next we perform a `filter()` operation using the broken filtering function.  
# No error will occur at this point due to Spark's use of lazy evaluation.
# 
# The `filter()` method will not be executed *until* an action operation is
# invoked on the DataFrame.  We will perform an action by using the `count()`
# method to return a list that contains all of the elements in this DataFrame.

# In[ ]:


def brokenTen(value):
    """
    This function raises a NameError, as variable 'val' does not exist; the parameter is 
    named 'value'.
    """
    if (val < 10):
        return True
    else:
        return False

btUDF = udf(brokenTen)
brokenDF = subDF.filter(btUDF(subDF.age) == True)


# To see the error, we need to call an action!

# In[ ]:


# Now we'll see the error
# Scroll through the message
brokenDF.count()


# ## (c) Finding the bug
# 
# When the `filter()` method is executed, Spark calls the UDF. Since our UDF has
# an error in the underlying filtering function `brokenTen()`, an error occurs.
# 
# Scroll through the output "Py4JJavaError     Traceback (most recent call last)"
# part of the cell and first you will see that the line that generated the error
# is the `count()` method line. There is *nothing wrong with this line*. However,
# it is an action and that caused other methods to be executed. Continue scrolling
# through the Traceback and you will see the following error line:
# 
# `NameError: global name 'val' is not defined`
# 
# Looking at this error line, we can see that we used the wrong variable name in
# our filtering function `brokenTen()`.
# 
# ## (d) Moving toward expert style
# 
# As you are learning Spark, I recommend that you write your code in the form:
# 
# <pre>    
# df2 = df1.transformation1()
# df2.action1()
# 
# df3 = df2.transformation2()
# df3.action2()
# </pre>
# 
# 
# Using this style will make debugging your code much easier as it makes errors
# easier to localize - errors in your transformations will occur when the next
# action is executed.
# 
# Once you become more experienced with Spark, you can write your code with the
# form: 
# 
# <pre>
# df.transformation1()\
#   .transformation2()\
#   .action()
# </pre>
# 
# 
# 
# ##  (e) Readability and code style
# 
# Spark style is also to use `lambda` functions instead of separately defined functions,
# when their use improves readability and conciseness.

# In[ ]:


# Cleaner code through lambda use
myUDF = udf(lambda v: v < 10)
subDF.filter(myUDF(subDF.age) == True)


# To make the expert coding style more readable, enclose the statement in
# parentheses and put each method, transformation, or action on a separate line.

# In[ ]:


# Final version
from pyspark.sql.functions import *
dataDF.filter(dataDF.age > 20)      .select(concat(dataDF.first_name, lit(' '), dataDF.last_name).alias('full_name'), dataDF.occupation)      .show(10)


# # Behind the scenes (optional)
# 
# When you use DataFrames or Spark SQL, you are building up a _query plan_. Each
# transformation you apply to a DataFrame adds some information to the query plan.
# When you finally call an action, which triggers execution of your Spark job,
# several things happen:
# 
# 1. Spark's Catalyst optimizer analyzes the query plan (called an _unoptimized
# logical query plan_) and attempts to optimize it. Optimizations include (but
# aren't limited to) rearranging and combining `filter()` operations for
# efficiency, converting `Decimal` operations to more efficient long integer
# operations, and pushing some operations down into the data source (e.g., a
# `filter()` operation might be translated to a SQL `WHERE` clause, if the data
# source is a traditional SQL RDBMS). The result of this optimization phase is an
# _optimized logical plan_.
# 2. Once Catalyst has an optimized logical plan, it then constructs multiple
# _physical_ plans from it. Specifically, it implements the query in terms of
# lower level Spark RDD operations.
# 3. Catalyst chooses which physical plan to use via _cost optimization_. That is,
# it determines which physical plan is the most efficient (or least expensive),
# and uses that one.
# 4. Finally, once the physical RDD execution plan is established, Spark actually
# executes the job.
# 
# You can examine the query plan using the `explain()` function on a DataFrame. By
# default, `explain()` only shows you the final physical plan; however, if you
# pass it an argument of `True`, it will show you all phases.
# 
# (If you want to take a deeper dive into how Catalyst optimizes DataFrame
# queries, this blog post, while a little old, is an excellent overview: [Deep
# Dive into Spark SQL's Catalyst Optimizer](https://databricks.com/blog/2015/04/13
# /deep-dive-into-spark-sqls-catalyst-optimizer.html).)
# 
# Let's add a couple transformations to our DataFrame and look at the query plan
# on the resulting transformed DataFrame. Don't be too concerned if it looks like
# gibberish. As you gain more experience with Apache Spark, you'll begin to be
# able to use `explain()` to help you understand more about your DataFrame
# operations.

# In[ ]:


newDF = dataDF.distinct().select('*')
newDF.explain(True)

