def merge_and_label_sets(train_df, test_df):
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df

def add_family_size_column(df):
    # Add a "Family Size" column by combining "SibSp", "Parch", and adding 1
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df

def add_family_size_to_train(train_df):
    # Add a "Family Size" column to train_df by combining "SibSp", "Parch", and adding 1
    train_df["Family Size"] = train_df["SibSp"] + train_df["Parch"] + 1
    return train_df

def assign_age_interval(df):
    df["Age Interval"] = 0.0
    df.loc[df['Age'] <= 16, 'Age Interval'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[df['Age'] > 64, 'Age Interval'] = 4

    return df

def train_df(df): 
 train_df = assign_age_interval(train_df)
 return df


def assign_fare_interval(df):
    df['Fare Interval'] = 0.0
    df.loc[df['Fare'] <= 7.91, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare Interval'] = 2
    df.loc[df['Fare'] > 31, 'Fare Interval'] = 3

    return df


def train_df(df):
   train_df = assign_fare_interval(train_df)
   return df


def create_sex_pclass_column(df):
    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    return df

def process_all_df(df):
    process_all_df = create_sex_pclass_column(all_df)
    return df


def parse_names(row):
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")
    

def apply_name_parsing(df):
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(lambda row: parse_names(row), axis=1)
    return df


def train_df(df):
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(lambda row: parse_names(row), axis=1)
    return df


def add_family_type(datasets):
    # Iterate over the list of datasets and create the "Family Type" column
    for dataset in datasets:
        dataset["Family Type"] = dataset["Family Size"]
    return datasets

def classify_family_type(all_df, train_df):
    for dataset in [all_df, train_df]:
        dataset.loc[dataset["Family Size"] == 1, "Family Type"] = "Single"
        dataset.loc[(dataset["Family Size"] > 1) & (dataset["Family Size"] < 5), "Family Type"] = "Small"
        dataset.loc[dataset["Family Size"] >= 5, "Family Type"] = "Large"
    return dataset

def unify_titles(datasets):
    """
    This function standardizes the Titles column in the datasets by unifying different forms of titles.
    It replaces specific titles with standard ones and groups rare titles under 'Rare'.
    """
    for dataset in datasets:
        # Unify `Miss`
        dataset['Titles'] = dataset['Titles'].replace(['Mlle.', 'Ms.'], 'Miss.')
        
        # Unify `Mrs`
        dataset['Titles'] = dataset['Titles'].replace('Mme.', 'Mrs.')
        
        # Unify rare titles
        rare_titles = ['Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.', 
                       'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.']
        dataset['Titles'] = dataset['Titles'].replace(rare_titles, 'Rare')
    
    return datasets


def calculate_mean_survival(df):
    return df[['Titles', 'Sex', 'Survived']].groupby(['Titles', 'Sex'], as_index=False).mean()
