import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportions_ztest
import random

data_frame = pd.read_csv("crime.csv")
data_frame = data_frame.drop([
    'INCIDENT_ID',
    'OFFENSE_ID',
    'OFFENSE_CODE',
    'OFFENSE_CODE_EXTENSION',
    "OFFENSE_TYPE_ID",
    "INCIDENT_ADDRESS",
    "REPORTED_DATE",
    "IS_TRAFFIC",
    'IS_CRIME'],axis=1).dropna()

def drop_extraneous_crimes(data_frame,crime_list):
    for crime in crime_list:
        data_frame = data_frame[data_frame.OFFENSE_CATEGORY_ID != crime]
    return data_frame

data_frame = drop_extraneous_crimes(data_frame,["traffic-accident",
                                                "all-other-crimes",
                                                "other-crimes-against-persons",
                                                "white-collar-crime",
                                                "aggravated-assault",
                                                "robbery",
                                                "drug-alcohol",
                                                "arson",
                                                "murder"])

def discard_rows(data_frame):
    value_counts = pd.value_counts(data_frame['OFFENSE_CATEGORY_ID'])
    print(value_counts)
    minimum_category = min(value_counts)
    for category,key in zip(value_counts,value_counts.keys()):
        data_frame = data_frame.reset_index(drop=True)
        difference = category - minimum_category
        category_rows = data_frame.index[data_frame['OFFENSE_CATEGORY_ID']==key].tolist()
        rows_to_discard = random.sample(category_rows,difference)
        data_frame = data_frame.drop(index=rows_to_discard,axis=0)
    return data_frame

data_frame = discard_rows(data_frame)
unique_offenses = data_frame.OFFENSE_CATEGORY_ID.unique()
label_encoder = LabelEncoder().fit(unique_offenses)
data_frame['OFFENSE_CATEGORY_ID'] = label_encoder.transform(data_frame['OFFENSE_CATEGORY_ID'])

def standardize_time(date_column):
    quantified_dates = []
    for date in date_column:
        split_date = date.split(" ")
        time = split_date[1]
        time_components = time.split(":")
        quantified_date = float(time_components[0])*100 + float(time_components[1])
        day_partition = split_date[2]
        if day_partition == "PM":
            quantified_date += 1200
        quantified_dates.append(quantified_date)
    return quantified_dates

data_frame["FIRST_OCCURRENCE_DATE"] = standardize_time(data_frame['FIRST_OCCURRENCE_DATE'])
data_frame["LAST_OCCURRENCE_DATE"] = standardize_time(data_frame["LAST_OCCURRENCE_DATE"])

def convert_to_float(data_frame,feature_list):
    for feature in feature_list:
        data_frame[feature] = pd.to_numeric(data_frame[feature])
    return data_frame

data_frame = convert_to_float(data_frame, ["GEO_X","GEO_Y","GEO_LON","GEO_LAT","DISTRICT_ID","PRECINCT_ID"])

dummy_columns = pd.get_dummies(data_frame['NEIGHBORHOOD_ID'])
data_frame = data_frame.drop(["NEIGHBORHOOD_ID"],axis=1)
labels = np.array(data_frame['OFFENSE_CATEGORY_ID'])
data_frame = data_frame.drop(["OFFENSE_CATEGORY_ID"],axis=1)

data_frame = pd.concat([data_frame, dummy_columns], axis=1, sort=False)

feature_vectors = np.array(data_frame)
scaler = StandardScaler()
feature_vectors = scaler.fit_transform(feature_vectors)

X_train, X_test, y_train, y_test = train_test_split(feature_vectors,labels,test_size=0.3)

model = LogisticRegression(C=0.0001,max_iter=1000,multi_class='multinomial').fit(X_train,y_train)


success_proportion = model.score(X_test,y_test)
guessing_randomly_proportion = 0.2 

sample_size = len(X_test)

sample_successes_a = int(sample_size*success_proportion)
sample_successes_b = int(sample_size*guessing_randomly_proportion)

successes = np.array([sample_successes_a,sample_successes_b])
sample_sizes = np.array([sample_size,sample_size])

stat, p_value = proportions_ztest(count=successes, nobs=sample_sizes,  alternative='larger')

print(success_proportion,guessing_randomly_proportion,sample_size)
print(stat,p_value)


