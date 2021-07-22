import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Imports for Numerical Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#Categorical Columns Preprocessing Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#drop address and area columns temporarily to get initial model out
filter_columns = ["Latitude", "Longitude", "DOM", "Tot BR", "Tot Baths", "TotFlArea", "Age", 
                  "Frontage - Feet", "#Kitchens", "List Date", "Sold Date",
                  "2020 Total Value", "2020 Land Value", "2020 Buildings Value", 
                  "2019 Total Value", "2019 Land Value", "2019 Buildings Value", 
                  "Interest Rate", "Season", "S/A", "TypeDwel", "Showing Appts"]

numerical_columns = ["Latitude", "Longitude","DOM", "Tot BR", "Tot Baths", "TotFlArea", "Age", 
                     "Frontage - Feet", "#Kitchens", "2020 Total Value", "2020 Land Value", 
                     "2020 Buildings Value", "2019 Total Value", "2019 Land Value", 
                     "2019 Buildings Value",  "Interest Rate"]

nominal_columns = ["S/A", "TypeDwel", "Showing Appts"]
ordinal_columns = ["Season"]
categorical_columns = ordinal_columns + nominal_columns
season_ordinal_encoding_order = ["Winter", "Spring", "Summer", "Fall"]

#returns a dataframe of data from a list of files (add proper comment here)
def load_data(dir, search_string):
  '''
  Returns a dataframe by combining all csv files in a directory whose name contains search_string
  (string, string) --> dataframe

  Argument Nmaes;
    dir: directory path
    search_string: string to search for in csv files' name
  '''
  #get all the data files in a list
  all_files_path = list_files(dir)
  data_files_path = []
  for file_path in all_files_path:
      if search_string in file_path and file_path.lower().endswith(".csv"):
          data_files_path.append(file_path)
  
  #load the data into dataframe and return the dataframe
  df = df_combined(data_files_path)
  #reset index and drop redundant index column
  df.reset_index(inplace = True)
  df.drop(columns = 'index', inplace = True)
  return df

def list_files(dir):
  '''
  Returns a list of paths in a directory
  (string) --> list of string

  Argument Names:
    dir: the directory path
  '''
  r = []
  for root, dirs, files in os.walk(dir):
      for name in files:
          r.append(os.path.join(root, name))
  return r

#return a data frame by combining list of csv files
def df_combined (csv_files_path):
  '''
    return a data frame by combining list of csv files om csv_files_path
    (list of string) ---> dataframe
    Argument Names:
      csv_files_path: list of csv files path
  '''
  df_all = pd.DataFrame()
  
  for fls_name in csv_files_path:
      df1 = pd.DataFrame ()
      if os.stat (fls_name).st_size != 0: #skip empty files as those causes errors
          df1 = pd.read_csv (fls_name , encoding = 'utf-8-sig', header = 0, skip_blank_lines= True) #read a file into dataframe
          if (fls_name  == csv_files_path[0]) or (df_all.empty): #combines the dataframes
              if not (df1.empty):
                  df_all = df1
          else:
              if not (df1.empty):
                  df_all = df_all.append (df1)
  
  return df_all

from enum import Enum
class Sampling(Enum):
    Random = 1
    Stratified = 2


def create_train_test_set(df, target_col_name = "Sold Price", strategy = Sampling.Stratified, stratified_cat_col_name = "S/A"):
  '''
  Create train and test set based on sampling strategy 
  (dataframe, string, Sampling enumeration, string) ---> (dataframe, series, dataframe, series)

  Argument Names:
    df: the dataframe containing the data
    target_col_name: the target column
    strategy: sampling strategy (Random/Stratified)
    stratified_cat_col_name: column to use for stratified sampling
  '''
  
  #reset index or otherwise causes issue with Stratified sampling
  df.reset_index(inplace = True)
  df.drop(columns = 'index', inplace = True)
  
  if strategy == Sampling.Random:
      from sklearn.model_selection import train_test_split
      train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
      
      df_train = train_set.drop(target_col_name, axis = 1)
      df_train_labels = train_set[target_col_name].copy()
      
      df_test = test_set.drop(target_col_name, axis = 1)
      df_test_labels = test_set[target_col_name].copy()

      return df_train, df_train_labels, df_test, df_test_labels
  
  if strategy == Sampling.Stratified:
      from sklearn.model_selection import StratifiedShuffleSplit
      #creates a stratified shuffle split object
      split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 

      #we generate train indices and test indices based on stratified_cat_col_name and we obtain the dataset into two dataframe
      for train_index, test_index in split.split(df, df[stratified_cat_col_name]):
          strat_train_set = df.loc[train_index]
          strat_test_set = df.loc[test_index]
          
          df_train = strat_train_set.drop(target_col_name, axis = 1)
          df_train_labels = strat_train_set[target_col_name].copy()
          
          df_test = strat_test_set.drop(target_col_name, axis = 1)
          df_test_labels = strat_test_set[target_col_name].copy()

          return df_train, df_train_labels, df_test, df_test_labels

#All the fit methods of all the classes return self. So I won't comment on them
#In addition, the object is always passed in so as an input so I won't comment on it either
#Unless otherwise stated
  # the X parameter in the fit and transform method is a dataframe
  #the output of fit method is the object itself
  #the output of transform method is a dataframe
class RemoveDuplRows(BaseEstimator, TransformerMixin):
  '''
    Removes duplicate rows from a dataframe
  '''
  def __init__(self, col_name):
    '''
      col_name: the name of the column which to use to remove duplicates from a dataframe
    '''
    self.col_name = col_name
  
  def fit(self, X):
    return self
  
  def transform(self, X):
    X.drop_duplicates(subset = self.col_name, inplace = True)
    return X

class MultiJoinColumns(BaseEstimator, TransformerMixin):
  '''
  Joins mulitple dataframes using specified joining columns
  #n dataframe to join
  #n - 1 columns to join on
  '''
  def __init__(self, join_on_cols_names):
    '''
    Argument names:
      join_on_cols_names: list of column names which are to use for joining dataframes
    '''
    self.join_on_cols_names = join_on_cols_names

  def fit(self, dfs):
    '''
    dfs: list of dataframes
    '''
    return self
  
  def transform(self, dfs):
    '''
    (dataframes) --> (dataframe)
    dfs: list of dataframes
    '''
    final_dataframe = pd.DataFrame()
    for i in range(len(self.join_on_cols_names)):
        if i == 0:
            final_dataframe = pd.merge(left = dfs[i], right = dfs[i + 1], on = self.join_on_cols_names[i])
        else:
            final_dataframe = pd.merge(left = final_dataframe, right = dfs[i + 1], on = self.join_on_cols_names[i])
    return final_dataframe

#return a dataframe with specified coloumns (Working)
class ExtractSelCol(BaseEstimator, TransformerMixin):
  '''
  Return a dataframe with specified columns only
  '''
  def __init__(self, col_names):
    '''
    col_names: list of column names which are to be preserved in the passed in dataframe
    '''
    self.col_names = col_names
  
  def fit(self, X):
    return self
  
  def transform(self, X):
    return X[self.col_names]

#filter dataframe by rows
#Transformer #1
class ExtractSelRow(BaseEstimator, TransformerMixin):
  '''
  Return a dataframe where specified column has specified value
  '''
  def __init__(self, col_name, col_value_to_select, inverse = False):
    '''
    col_name: the name of the column which is to be used for filtering
    col_values_to_select: the value which to use for filtering
    inverse: False if want to return the rows that contain col_values_to_select. True if you want the inverse result
    '''
    self.col_name = col_name
    self.col_value_to_select = col_value_to_select
    self.inverse = inverse
  
  def fit(self, X):
    return self
  
  def transform(self, X):
    if not self.__dict__['inverse']: #avoid infinite recursion by using dict attribute instead of self.inverse
        return X.loc[X[self.col_name] == self.col_value_to_select, :] 
    else:
        return X.loc[~(X[self.col_name] == self.col_value_to_select), :]

#Drop Address and Area Column and replace them with Latitude and Longitude (Working)
class AddressLatitudeLongitude(BaseEstimator, TransformerMixin):
  '''
  Create Latitude and Longitude Column from Address Column
  '''
  def __init__(self, address_col_name, area_col_name, country = 'Canada'):
    '''
    address_col_name: name of the column containing address
    area_col_name: name of the column containing area
    country: country name
    '''
    self.address_col_name = address_col_name
    self.area_col_name = area_col_name
    self.country = country

  def fit(self, X):
      return self

  def transform(self, X):
    X = X.copy()
    #calculate longitude and latitude
    locations = list(map(location_latitude_longitude, X[self.address_col_name], X[self.area_col_name]))
    latitude = pd.Series([location[0] for location in locations], index = X.index, name = "Latitude")
    longitude = pd.Series([location[1] for location in locations], index = X.index, name = "Longitude")
    #drop address and area from X and append latitude and longitude to the end
    X.drop(columns = [self.address_col_name, self.area_col_name], inplace = True)
    return pd.concat([latitude, longitude, X], axis = 1)

import geopy
service = geopy.Nominatim(user_agent = "myGeocoder")

def location_latitude_longitude(street, city, country = 'Canada'):
  '''
  Returns latitude and longitude of an property listing
  (string, string, string) ---> (float, float)
  
  Argument names:
    street: the address of the property
    city: the city the address in
    country: the country the address in
  '''
  location_query = {'street': street, 'city': city, 'country': country}
  #print(location_query)
  location_obj = service.geocode(location_query)
  if not location_obj is None: 
      return location_obj.latitude, location_obj.longitude
  else: #when address is invalid then use the city's longitude and latitude
      location_query = {'city': city, 'country': 'Canada'}
      location_obj = service.geocode(location_query)
      return location_obj.latitude, location_obj.longitude

from geopy import distance
def distance_btw_locations(lat1, long1, lat2, long2):
  '''
  Returns the distance between two locations in km
  (float, float, float, float) ---> float

  Argument Names:
    lat1: latitude of location 1
    long1: longitude of location 1
    lat2: latitude of location 2
    long2: longitude of lcoation 2

  '''
  return distance.distance((lat1, long1), (lat2, long2)).km

class AddressMergeLatitudeLongitude(BaseEstimator, TransformerMixin):
  '''
    Create Latitude and Longitude Column from Address Column. This class accomplishes same result as class AddressLatitudeLongitude
    But it uses previously generated latitude and longitude of every property and just merges the latitude and longitude
    column to passed in listing data
  '''
  def __init__(self, address_col_name, long_lat_file_path):
    '''
    address_col_name: name of the column containing addresses
    long_lat_file_path: the file containing latitude and longitude of each address (it contains three columns: Address, Latitude, Longitude)
    '''
    self.address_col_name = address_col_name
    self.long_lat_file_path = long_lat_file_path

  def fit(self, X):
    return self
  
  def transform(self, X):
    X = X.copy()
    #get longitude and latitude data and create a dataframe conssiting of address, area, latitude and longitude
    longitude_latitude_data = pd.read_csv(self.long_lat_file_path)
    longitude_latitude_merged_data = pd.merge(left = X[self.address_col_name], right = longitude_latitude_data, on = self.address_col_name)
    #set the index of longitude_latitude_merged_data to that of X to make sure concat operation goes through correctly
    longitude_latitude_merged_data.index = X.index
    #return a dataframe where address is dropped and replaced with longitude and latitude
    #X.drop(columns = [self.address_col_name], inplace = True) #TODO: MAYBE NEED TO uncomment this later on
    return pd.concat([longitude_latitude_merged_data[["Latitude", "Longitude"]], X], axis = 1)

class DatetimeToYearMonth(BaseEstimator, TransformerMixin):
  '''
  Transforms DateTime column and generates two additional columns: Year and Month
  '''
  def __init__(self, date_col_name, house_data_df_index):
    '''
    date_col_name: the name of the column containing the dates
    house_data_df_index: the index of dataframe in list of passed in dataframes
    '''
    self.date_col_name = date_col_name
    self.house_data_df_index = house_data_df_index

  def fit(self, dfs):
    return self
  
  def transform(self, dfs):
    '''
    (dataframes) ---> (dataframes)
    dfs: list of dataframes
    '''
    house_data = dfs[self.house_data_df_index]
    #make inplace change to to house_data; so don't need to reassign house_data back to list of dataframes
    house_data["Month"] = pd.to_datetime(house_data[self.date_col_name]).dt.month
    house_data["Year"] = pd.to_datetime(house_data[self.date_col_name]).dt.year
    #house_data.drop(columns = self.date_col_name, inplace = True); need sold date for calculating DOM later. So don't drop it yet
    return dfs

class SetGreaterValueNan(BaseEstimator, TransformerMixin):
  '''
  Set all the values in a numerical columns which values are greater than certain value to nan
  '''
  def __init__(self, col_name, col_value):
    '''
    col_name: the column name which to process
    col_value: the value which is to use as filter
    '''
    self.col_name = col_name
    self.col_value = col_value
  
  def fit(self, X):
    return self
  
  def transform(self, X):
    X = X.copy()
    X.loc[X[self.col_name] > self.col_value, self.col_name] = np.nan
    return X

class GetDom(BaseEstimator, TransformerMixin):
  '''
  Creates a new column in the dataframe "DOM" (day in market) which is SoldDate - ListDate
  '''
  def __init__(self, list_date_col_name, sold_date_col_name, dom_col_name):
    '''
    list_date_col_name: the name of the column containing list date
    sold_date_col_name: the name of the column containing sold date
    dom_col_name: the name of the column to use for the new column "DOM"
    '''
    self.list_date_col_name = list_date_col_name
    self.sold_date_col_name = sold_date_col_name
    self.dom_col_name = dom_col_name
  
  def fit(self, X):
    return self
  
  def transform(self, X):
    X = X.copy()
    dom_calculated = pd.to_datetime(X[self.sold_date_col_name]) - pd.to_datetime(X[self.list_date_col_name])
    dom_calculated = dom_calculated.map(convert_day_integer)
    #drop List Date and Sold Date and set DOM (manually entered) to DOM_Calculated
    X.drop(columns = [self.list_date_col_name, self.sold_date_col_name], inplace = True)
    X[self.dom_col_name] =  pd.Series(dom_calculated, index = X.index)
    return X

def convert_day_integer(delta_date):
  '''
  Returns the number of days in timedelta object
  (timedelta) ---> int
  Argument Name:
    delta_date: the timedelta object from which to extract the number of days from
  '''
  return delta_date.days

class CalcPriceSqft(BaseEstimator, TransformerMixin):
  '''
  Transforms property sold price to sold price / total floor area of property
  '''
  def __init__(self, price_col_name, floor_area_col_name):
    '''
    price_col_name: the name of the price column
    floor_area_col_name: the name of the column containing values for total floor area
    '''
    self.price_col_name = price_col_name
    self.floor_area_col_name = floor_area_col_name
  
  def fit (self, X):
    return self
  
  def transform(self, X):
    X[self.price_col_name] = X[self.price_col_name].astype('float') / X[self.floor_area_col_name].astype('float')
    return X

class NearestPropsData(BaseEstimator, TransformerMixin):
  '''
  Adds price_sq_ft of nearest n properties for each listing and cumulative distance (from subject property to n properties) and
  cumulative date difference (the number of days between the sold date of subject property to sold date of comparing property)
  '''
  def __init__(self, lat_col_name, long_col_name,  sold_date_col_name, sold_price_sq_ft_col_name, num_nearest_properties):
    '''
    lat_col_name: the name of the column containing latitude
    long_col_name: the name of the column containing longitude
    sold_date_col_name: the name of the column containing sold date
    sold_price_sq_ft_col_name: the name of the column containing sold price per square foot
    self.num_nearest_properties: the number of nearest property to look at
    '''
    self.lat_col_name = lat_col_name
    self.long_col_name = long_col_name
    self.sold_date_col_name = sold_date_col_name
    self.sold_price_sq_ft_col_name = sold_price_sq_ft_col_name
    self.num_nearest_properties = num_nearest_properties
  
  def fit(self, X):
    return self

  def transform(self, X):

    #for each listing, get the nearest n property data and the hold all nearest n property data for each property in a list
    X[self.sold_date_col_name] = pd.to_datetime(X[self.sold_date_col_name])
    all_nearest_property_data = []
    #X = X.tail(7) #temporary to speed up data generation process
    for i in X.index:
      #create a series where we have distance between current property to all other property
      lat1 = pd.Series(X.loc[i, self.lat_col_name]).repeat(X.shape[0]) #create a series with lat1 repeating X.shape[0] times
      long1 = pd.Series(X.loc[i, self.long_col_name]).repeat(X.shape[0]) #create a series with long1 repeating X.shape[0] times
      distance_other_props = pd.Series(map(distance_btw_locations, lat1, long1, X[self.lat_col_name], X[self.long_col_name]), index = X.index)

      #create a series where we have date differnce between current properties' sold date to other properties' sold date
      date_diff_other_props = (X[self.sold_date_col_name] - X.loc[i, self.sold_date_col_name]).dt.days.abs()

      #construct a dataframe with distance and date difference
      data = {'distance': distance_other_props, 'date_diff': date_diff_other_props}
      current_property_compare_data = pd.DataFrame(data = data) #index already present from series' so don't need to use index = X.index

      #sort the dataframe and extract the first n rows (n = number of nearest property data desired)
      current_property_compare_data.drop(index = i, inplace = True) #drwop the row corresponding to data to itself
      current_property_compare_data.sort_values(['distance', 'date_diff'], inplace = True)
      current_property_compare_data = current_property_compare_data.iloc[:self.num_nearest_properties, :]

      #join the price_sq of nearest n properties
      current_property_compare_data['Sold Price'] = X[self.sold_price_sq_ft_col_name] #should join properly using the indices of both series

      #flatten Sold Price to a row numpy row vector and add cumulative distance and cumulative date difference
      current_property_compare_data_transf = current_property_compare_data['Sold Price'].to_numpy()
      current_property_compare_data_transf = np.append(current_property_compare_data_transf, [current_property_compare_data['distance'].sum(), current_property_compare_data['date_diff'].sum()])

      #add current property's nearest properties data to master array
      all_nearest_property_data.append(current_property_compare_data_transf)

    #create a dataframe with all compare data so we can merge to X
    all_nearest_property_data_transf = pd.DataFrame(data = all_nearest_property_data, columns = nearest_property_data_columns, index = X.index)
    X = pd.merge(left = X, right = all_nearest_property_data_transf, left_index = True, right_index = True)
    return X

class RemoveBadChars(BaseEstimator, TransformerMixin):
  '''
  Remove list of bad characters from list of column names in a dataframe
  '''
  def __init__(self, cols_names, bad_chars):
    '''
    cols_names: list of column names from which to remove bad characters
    bad_chars: list of bad characters to remove
    '''
    self.cols_names = cols_names
    self.bad_chars = bad_chars
  
  def fit(self, X):
    return self
  
  def transform(self, X):
    for col_name in self.cols_names:
        for bad_char in self.bad_chars:
            #print(X[col_name].dtype)
            X[col_name] = X[col_name].astype('str')
            X[col_name] = X[col_name].str.replace(bad_char, "")
    return X

class ImputeNomColMode(BaseEstimator, TransformerMixin):
  '''
  Replaces missing values in nominal columns with its mode
  '''
  def __init__(self, cols_names):
    '''
    cols_names: list of columns for which we need to perform imputation
    '''
    self.cols_names = cols_names

  def fit(self, X):
    return self

  def transform(self, X):
    for col_name in self.cols_names:
        X.loc[X[col_name].isnull(), col_name] = X[col_name].value_counts().index[0]
    return X

class SimpleImputerCustom(BaseEstimator, TransformerMixin):
  '''
  Fill in missing values in numerical column(s) using the median strategy. Usage is similar to that of SimpleImputer from sklearn.impute. 
  '''
  def __init__(self, cols_names):
    '''
    Argument Names:
        cols_names list of columns for which we need to impute for
    '''
    self.cols_names = cols_names
  def fit(self, X):
      return self

  def transform(self, X):
    imputer = SimpleImputer(missing_values=np.nan, strategy = "median")
    X = X.copy()
    #Imputing Data
    #print(X[:, self.cols].shape)

    if len(self.cols_names) == 1:
        #create a duplicate of the same column so we get 2d array so we can use simple imputer
        #at the end the duplicate column will be discarded anyways
        data_to_impute = pd.DataFrame({self.cols_names[0] : X[self.cols_names[0]], self.cols_names[0] + "Dup": X[self.cols_names[0]]}).to_numpy()
    else:
        data_to_impute = X[self.cols_names].to_numpy()

    imputed_data = imputer.fit_transform(data_to_impute) #input needs to be 2d array

    #then do inplace replace in dataframe
    for col_idx in range(0, len(self.cols_names)):
        #print(self.cols_names[col_idx])
        X[self.cols_names[col_idx]] = pd.Series(imputed_data[:, col_idx], index = X.index)
    return X

class StandardScalerCustom(BaseEstimator, TransformerMixin):
  '''
  Scales numerical column(s) using StandardScaler from sklearn.preprocessing . Usage is same as in using StandardScaler. 
  '''
  def __init__(self, cols_names):
    '''
    Argument Names:
        cols_names: list of column names which we need to perform scaling for
    '''
    self.cols_names = cols_names

  def fit(self, X):
    return self

  def transform(self, X):
    standard_scaler = StandardScaler()
    data_to_scale = X[self.cols_names].to_numpy()
    #Scaling Data
    if len(self.cols_names) == 1:
        #create a duplicate of the same column so we get 2d array so we can use simple imputer
        #at the end the duplicate column will be discarded anyways
        data_to_scale = pd.DataFrame({self.cols_names[0] : X[self.cols_names[0]], self.cols_names[0] + "Dup": X[self.cols_names[0]]}).astype('float').to_numpy()
    else:
        data_to_scale = X[self.cols_names].astype('float').to_numpy()

    scaled_data = standard_scaler.fit_transform(data_to_scale) 
    for col_idx in range(0, len(self.cols_names)):
        #print(self.cols_names[col_idx])
        #print(scaled_data[:, col_idx])
        X[self.cols_names[col_idx]] = pd.Series(scaled_data[:, col_idx], index = X.index)
    return X

class OrdinalEncodingCustom(BaseEstimator, TransformerMixin):
  '''
  Encodes ordinal data. Usage is similar to that of OrdinalEncoder from sklearn.preprocessing
  '''
  def __init__(self, cols_names, categories):
    '''
    Argument Names:
        cols_names: name of columns which we need to encode
        categories: the categories argument for OrdinalEncoder
    '''
    self.cols_names = cols_names
    self.categories = categories

  def fit(self, X, y = None):
    return self
  
  def transform(self, X):
    to_encode_data = X[self.cols_names].copy().astype(object) #Need object dtype to use ordinal encoding
    ordinal_encoder = OrdinalEncoder(categories = self.categories)
    data_encoded = ordinal_encoder.fit_transform(X[self.cols_names].to_numpy())
    #Replace existing columns which are to be encoded with encoded values
    for i in range(len(self.cols_names)):
        X[self.cols_names[i]] = pd.Series(data_encoded[:, i], index = X.index)
    return X

one_hot_encoding_cols_catgs = []
class OneHotEncodingCustom(BaseEstimator, TransformerMixin):
  '''
  One hot encodes categorical data. Usage is similar to that of OneHotEncoder from sklearn.preprocessing
  '''
  def __init__(self, cols_names):
    '''
    cols_names: list of column names for which we need to perform one hot encoding
    '''
    self.cols_names = cols_names
  
  def fit(self, X, y = None):
    return self
  
  def transform(self, X): 
    one_hot_encoder = OneHotEncoder()

    #one hot encoding data
    one_hot_encoded_data = one_hot_encoder.fit_transform(X[self.cols_names].to_numpy()).toarray()
    
    #Get the column header for one-hot encoding; one hot encoder categories always gives list of list
    #Set it such that each column have the following format: Col_Name: Catg1, Col_Name: Catg2, and so on
    global one_hot_encoding_cols_catgs
    one_hot_encoding_cols_catgs = [] #reset the variable holding one hot encoding categories just in case
    for col_idx in range(0, len(self.cols_names)):
        for catgs in one_hot_encoder.categories_[col_idx]:
            one_hot_encoding_cols_catgs.append(self.cols_names[col_idx] + ": " + catgs)
        
    #Dropping columns that was encoded
    X.drop(columns = self.cols_names, inplace = True)

    #Append the one hot encoded data to dataframe and return the data
    one_hot_encoded_df = pd.DataFrame(data = one_hot_encoded_data, columns = one_hot_encoding_cols_catgs, index = X.index)

    return pd.concat([X, one_hot_encoded_df], axis = 1)

class ConvertFloat(BaseEstimator, TransformerMixin):
  '''
  Converts the datatype of an ndarray to float. This transformer should only be used
  after making sure that all the columns can be casted to float
  '''
  def __init__(self):
      pass
  def fit(self, X):
      return self
  def transform(self, X):
    X_float = np.empty(shape = (X.shape[0], X.shape[1]), dtype = float)
    for col_idx in range(0, X.shape[1]):
        #print(X.columns[col_idx])
        X_float[:, col_idx] = X[X.columns[col_idx]].to_numpy().astype(float)
    return X_float

#paths to important files
house_data_dir = r"C:\Users\KI PC\OneDrive\Documents\Software Engineering and Computer Science\Internships\Riipen - KnockNow\BC-House-Pricing-Model"
seasons_data_path = r"C:\Users\KI PC\OneDrive\Documents\Software Engineering and Computer Science\Internships\Riipen - KnockNow\BC-House-Pricing-Model\month_seasons.csv"
#mortgage rate from here: https://www.ratehub.ca/historical-mortgage-rates-widget
interest_rate_2020_path = r"C:\Users\KI PC\OneDrive\Documents\Software Engineering and Computer Science\Internships\Riipen - KnockNow\BC-House-Pricing-Model\2020_month_by_month_interest_rate.csv"
assesement_data_path = r"C:\Users\KI PC\OneDrive\Documents\Software Engineering and Computer Science\Internships\Riipen - KnockNow\BC-House-Pricing-Model\West-van-assessments.csv"
longitude_latitude_data_path = r"C:\Users\KI PC\OneDrive\Documents\Software Engineering and Computer Science\Internships\Riipen - KnockNow\BC-House-Pricing-Model\longitude_latitude_data.csv"

#number of nearest properties and the column names from nearest property data
num_nearest_properties = 5
nearest_property_data_columns = ['price_sq_ft{}'.format(i + 1) for i in range(num_nearest_properties)] + ['distance_tot', 'date_diff_tot']

#this pipeline gets initial data to be used for analysis
get_data = Pipeline([
    ('get_year_month', DatetimeToYearMonth("Sold Date", 0)), #use dataframe.copy() in transform method to break the link to dataframe being passed
    ('join_data_columns', MultiJoinColumns(["Address", "Month", "Month"])),
    ('remove_duplicate_listings', RemoveDuplRows("ML #")),
    ('filter_by_sold', ExtractSelRow("Status", "S")),
    ('filter_by_2020', ExtractSelRow("Year", 2020)),
    ('filter_zero_2020_prop_value', ExtractSelRow("2020 Total Value", 0, True)),
    ('remove_bad_char', RemoveBadChars(["Sold Price", "TotFlArea"], [",", "$"])),
    ('calc_price_sqft', CalcPriceSqft("Sold Price", "TotFlArea")),
    ('latitude_longitude', AddressMergeLatitudeLongitude("Address", longitude_latitude_data_path)),
    ('nearest_prop_data', NearestPropsData("Latitude", "Longitude", "Sold Date", "Sold Price", num_nearest_properties)),
    ('filter_col', ExtractSelCol(filter_columns + nearest_property_data_columns + ["Sold Price"])) #moved here as need the area column to get nearest property data; might move all filtering to bottom
    ])

#this pipeline transforms data into a format required for ML algorithms
data_preprocessing_pipeline  = Pipeline([
    ('calculate_dom', GetDom("List Date", "Sold Date", "DOM")),
    ('invalid_age_nan', SetGreaterValueNan("Age", 150)),
    ('replace_char_fl_area', RemoveBadChars(["TotFlArea"], [","])),
    ('impute_num_col', SimpleImputerCustom(numerical_columns)),
    ("std_scalar", StandardScalerCustom(numerical_columns + nearest_property_data_columns)),
    ("impute_nom_col_mode", ImputeNomColMode(nominal_columns)),
    ('ordinal_encoding', OrdinalEncodingCustom(ordinal_columns, categories = [season_ordinal_encoding_order])),
    ('one_hot_encoding', OneHotEncodingCustom(nominal_columns)),
    ('convert_float', ConvertFloat())
    ])

if __name__ == "__main__":
  #get data
  house_data = load_data(house_data_dir, "Spreadsheet")
  assesement_data = pd.read_csv(assesement_data_path)
  seasons_data = pd.read_csv(seasons_data_path)
  interest_rate_2020_data = pd.read_csv(interest_rate_2020_path)
  house_data_sold = get_data.fit_transform([house_data, assesement_data, seasons_data, interest_rate_2020_data])
  house_data_sold.to_csv("data_V3.csv") #export to csv if want to save time later on
  
  #get train and test data
  house_data_sold_train, house_data_sold_labels, house_data_sold_test, house_data_sold_labels =  create_train_test_set(df = house_data_sold)
  
  #second pipeline here to transform the data
  house_data_sold_train_prepared = data_preprocessing_pipeline.fit_transform(house_data_sold_train)