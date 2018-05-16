# FORECASTING OF ORDER DEMAND IN WAREHOUSES USING AUTOREGRESSIVE INTEGRATED MOVING AVERAGE

Authors:  **VIJAY GANESH SRINIVASAN**, **SOHIT REDDY KALLURU**, **RAMAKRISHNA POLEPEDDI**.

YouTube Video:  [Link](http://your_link_goes_here)

---

**NOTE**:  For more details on the statistical understanding of the project kindly read *Introduction to time series and forecasting
Book by Peter J. Brockwell and Richard A. Davis, Production and Operations Analysis by Steven Nahmias and Supply Chain Engineering: Models and Applications by A. Ravi Ravindran, Donald P. Warsing, Jr.* 


---

## FORECASTING

![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/Forecasting_Title_Image.PNG)
---


## Project outline
- The objective of the project is to forecast the order demand using AUTOREGRESSIVE INTEGRATED MOVING AVERAGE model for 4 warehouses respectively.
- For this analysis we have downloaded the data from Kaggle. (https://www.kaggle.com/felixzhao/productdemandforecasting/data)
- The basics of ACF, PACF, rolling mean average, rolling standard deviation and correlogram are explained in this documentation.
- By the end of the documentation you'll have a clear idea about **A**utoregressive **I**ntegrated **M**oving **A**verage or **ARIMA** model, data visualization, data analysis, statistical library functions in python and creation of interactive plots using plotly.  

---

## What makes the dataset intersting
- The dataset contains historical product demand for a manufacturing company with footprints globally. 
- The company provides thousands of products within dozens of product categories for 7 years. There are four central warehouses to ship products within the region it is responsible for.
- The data is available in the .csv format which allows us to perform the dataframe operations easily.

---

### NOTE : Packages to install before running this program

### Plotly - Modern Visualization for the Data Era

- It is one important package to be installed to have interactive plots. It is very easy to use.

### Installing instructions

- To install Plotly's python package, use the package manager pip inside your terminal.

```
$ pip install plotly 
```
- After installing plotly run python and configure plotly by entering your credentials.

```
import plotly
plotly.tools.set_credentials_file(username='MyAccount', api_key='********')
```
- Use this hyperlink to create https://plot.ly/feed an account and to generate API key follow the instructions mentioned in the website. 

---

## 5 Steps towards Forecasting

![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/5_Steps_Towards_Forecasting.PNG)
---

## Introduction to ARIMA

- ARIMA is a forecasting technique. ARIMA– Auto Regressive Integrated Moving Average the key tool in Time Series Analysis.
- Models that relate the present value of a series to past values and past prediction errors - these are called ARIMA models.
- ARIMA models provide an approach to time series forecasting. 
- ARIMA is a forecasting technique that projects the future values of a series based entirely on its own inertia.
- Exponential smoothing and ARIMA models are the two most widely-used approaches to time series forecasting. 
- Exponential smoothing models are based on a description of trend and seasonality in the data, ARIMA models aim to describe the autocorrelations in the data.
- Its main application is in the area of short term forecasting requiring at least 40 historical data points. 
- It works best when your data exhibits a stable or consistent pattern over time with a minimum amount of outliers.
- ARIMA is usually superior to exponential smoothing techniques when the data is reasonably long and the correlation between past observations is stable.

--- 

## Program explanation

### Code summary

## References
*In this section, provide links to your references and data sources.  For example:*
- Source code was adapted from [the magic source code farm](http://www.amagicalnonexistentplace.com)
- The code retrieves data from [the organization for hosting cool data](http://www.anothermagicalnonexistentplace.com)
- The forecasting models were modified from [some academic research paper](http://www.linktotheacademicpaperyouused.com)


## Explanation of the Code
*In this section you should provide a more detailed explanation of what, exactly, your Python script(s) actually do.  Your classmates should be able to read your explanation and understand what is happening in the code.*

The code, `needs_a_good_name.py`, begins by importing necessary Python packages:
```
import matplotlib.pyplot as plt
```

- *NOTE:  If a package does not come pre-installed with Anaconda, you'll need to provide instructions for installing that package here.*

We then import data from [insert name of data source].  We print the data to allow us to verify what we've imported:
```
x = [1, 3, 4, 7]
y = [2, 5, 1, 6]

for i in range(0,len(x)):
	print "x[%d] = %f" % (i, x[i])		
```
- *NOTE 1:  This sample code doesn't actually import anything.  You'll need your code to grab live data from an online source.*  
- *NOTE 2:  You will probably also need to clean/filter/re-structure the raw data.  Be sure to include that step.*

Finally, we visualize the data.  We save our plot as a `.png` image:
```
plt.plot(x, y)
plt.savefig('samplefigure.png')	
plt.show()
```

The output from this code is shown below:

![Image of Plot](images/samplefigure.png)

---

## How to Run the Code
*Provide step-by-step instructions for running the code.  For example, I like to run code from the terminal:*
1. Ensure that you have registered for the [insert name of API] API key.  (You may reference the instructions for doing this.)

2. Ensure that you have installed necessary Python packages. (Again, you may include a reference here to a prior section in the README that provides the instructions.)


2. Open a terminal window.

2. Change directories to where `needs_a_good_name.py` is saved.

3. Type the following command:
	```
	python needs_a_good_name.py
	```

- *NOTE: You are welcome to provide instructions using Anaconda, IPython, or Jupyter notebooks.*

---

## Results from your Analysis
*Last, but not least, you need to demonstrate your results.  You should include figures and/or tables of results.  What worked well?  What could be improved?*
