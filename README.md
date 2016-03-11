# ArnholdInstituteDataChallenge

The Python source code in the repository requires the following special libraries:

pandas v0.17.1, for using DataFrames to structure data and manipulate it easily

scikit-learn v0.17, for using Python-standard machine learning algorithms

smopy v0.0.3, for displaying images of data overlaid on maps of locations

geopy v1.11.0, for getting latitude/longitude data on an English-language mailing address


and the libraries from installing the anaconda client v1.2.1.



The source code includes the following files:

cleanLead.py - this reads the BLL and building condition data from multiple files and stores them in a single pandas dataframe, as well as getting the lat/long coordinates for each UHF (United Hospital Fund) neighborhood, and then stores the dataframe in a pickle file that another python script can access.

exploreLead.py - this creates new fields in the single dataframe describing which borough a given data row is located in,  and then generates the exploratory graphs on the BLL and building condition data. It then performs predictive analytics using non-regularized linear regression and L1-regularized linear regression, and tries out k-means clustering on the data with 3 clusters.

cleanCensus.py - this isn't finished, but begins to convert New York City's Community Districts (CDs) to their corresponding UHF neighborhoods. This was done by first manually transcribing US Census data from Excel files into a CSV file, and then using the Python script to directly map CDs to PUMAs (Public Use Microdata Areas), which are statistical areas defined by the US Census Bureau. It then converts each PUMA to multiple ZCTAs (Zip Code Tabulation Areas), which are approximations of ZIP codes used by the Census Bureau, and then each ZCTA to multiple ZIP codes (which may overlap). This is not as easy as one might expect, since ZIP codes are not necessarily bound by latitude and longitude and instead follow addresses along streets. Much of this work involved reading PDFs or Excel files and then hand-coding dictionaries to convert data from one format to another, and meant that one CD could fit into multiple UHF neighborhoods. This script will be finished, and the new data will be incorporated into an updated version of this project which will be uploaded into another github repository.
