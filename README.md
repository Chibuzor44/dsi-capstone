# Quantifying customers dining experience with NLP

## Galvanize Data Science Immersive Capstone (In Progress)
<br>
#### Motivation
<p>
  A study from Cornell University conducted in 2005 <a href="https://www.researchgate.net/publication/237835565_Why_Restaurants_Fail"><i>Why restaurants fail</i><a/> found that 26.16 percent of independent restaurants failed during the first year of operation. Another recent study in 2014 <a href="https://www.researchgate.net/publication/267695784_Only_the_Bad_Die_Young_Restaurant_Mortality_in_the_Western_US"><i>Only the Bad Die Young: Restaurant Mortality in the Western US</i></a> also found that the restaurant franchize have actually improved a bit to 17 percent of independently owned full-service restaurant startups failing in their first year. One of the factors that affect retaurants from being successful is the inability to identify aspects of their services impacting customers satisfaction.
</p>
#### Overview
<p>
The availability of online review web services have made it relatively easy for restaurant patrons to get information about a retaurant before deciding to dinning there. Yelp is a popular review online service for restaurants and other businesses. This presents an opportunity for these restaurants to use these feedbacks to measure their customer satisfaction performance and improve on poorly performed aspects of their services. Most restaurants rely on star ratings to measure customer satisfaction, but star rating does not necessarily reveal the cause of a customer's satisfaction/dissatisfaction. This project focuses on using natural language processing (NLP) to analyze user reviews text to extract aspects of restaurants services that impact its customers.<br>
In other to get anticipated results out of this modelling, it should answer the following questions:
  <ul>
    <li>
      What are the factors responsible for a customers dining experience falling on the far opposite ends of satisfaction ?
    </li>
    <li>
      What stops a customer from who gives a 4-star rating from giving a 5-star ?
    </li>
    <li>
      What distinguishes a great restaurants service from a 3-star restaurants service ?
    </li>
    <li>
      How does the tone in review text change across review star ratings ?
    </li>
  </ul>
  ![image](/images/image1.png), ![image](/images/image2.png), ![image](/images/image3.png)
</p>

#### Datasets
<p>
Three sets of data was collected from <a href="https://www.yelp.com/dataset/challenge">yelp</a>, which is a publicly available dataset. The datasets are as follows:
  <ul>
  <li>Business dataset that describes all sorts of businesses that provide services to the general public. The            dataset has 174,567 records.</li>
  <li>Review dataset is a records customer reviews of differnt business in the business dataset collected. The dataset holds 5,261,669 records.</li>
  <li>User dataset holds the record of users who have given reviews and businesses they reviewed. The user data has 1,326,101 records.</li>
  </ul>
  The datasets were stored in AWS s3 bucket, and due to the size of the data, I was unable to do the computation on a local machine. I used AWS SageMaker instance running on EC2 and Elastic File Storage(EFS) for most of the data munging and model training.
  
#### Methodology
##### Data Cleaning
<p>
  <ul>
    <li>Step 1: Joining data sources. <br>
    The The three categories of data had some feature name that were nnot unique to it, and had to be renamed. The datasets had reviews records from both the US and other countries and had to be filtered to retain reviews for restaurants in US states alone and all three tables were joined on *business_id and user_id*. Exploring the data further shows 95.91 percent of the US data came from five states. I used the data for the five states for modelling and validation and the rest of the 4.09 percent of the data for testing the web app which I will discuss in a later section. This process can be seen in ![sage_yelp.ipynb](sage_yelp.ipynb).
![Data by states](/images/state.png)
  </li>
  <li>Step 2. Tokenizer, Lemmatize, Regex. <br>
    I used two functions * clean_stem(corpus) * and *regex(word)* in ![util.py](util.py) to tokenize each review into list of words, remove stop words as they do not communicate important information about the context of the review and remove strings that are digits or have digits in them and lemmatize the words. This returns a corpus of documents ready to be vectorized.  
  </li>
  <li>Step 3. Term Frequency Inverse Document Frequency (TFIDF). <br>
  Before building a model, the reviews have to be transformed into numerical representation. I used the TFIDF to turn the corpus into word vectors. 
  </li>
</ul>
</p>
##### Models
<p>
  To answer the questions posed in the overview, I used Logistic Regression to model the data by 1-star vs all and 5-star vs all.
  <ul>
  <li>Model_12</li>
  <li>Model_13</li>
  <li>Model_14</li>
  <li>Model_15</li>
  <li>Model_52</li>
  <li>Model_53</li>
  <li>Model_54</li>
  </ul>
</p>

#### Results
<p>
  
</p>
#### Conclusion
<p>
Every business is interested in improving their service delivery to keep customers happy all the time. Any business will continue to grow with service improvements but only if they know the areas of improvement. This method can be applied not only to restaurants, but any business that users leave reviews about the business' performance.
</p>
