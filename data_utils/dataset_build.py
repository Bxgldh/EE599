import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

def generate_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = {data_point["sentiment"]}
            """.strip()

def generate_test_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = 
            """.strip()

def load_and_split_data(path, train_size=300, test_size=300, seed=42):
        
    df = pd.read_csv(path, 
                    names=["sentiment", "text"],
                    encoding="utf-8", encoding_errors="replace")
    
    X_train = list()
    X_test = list()

    for sentiment in ["positive", "neutral", "negative"]:
        train, test  = train_test_split(df[df.sentiment==sentiment], 
                                        train_size=300,
                                        test_size=300, 
                                        random_state=42)
        X_train.append(train)
        X_test.append(test)
    
    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_test = pd.concat(X_test)

    eval_idx = [idx for idx in df.index if idx not in list(X_train.index) + list(X_test.index)]
    X_eval = df[df.index.isin(eval_idx)]
    X_eval = (X_eval
            .groupby('sentiment', group_keys=False)
            .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))
    X_train = X_train.reset_index(drop=True)

    X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), 
                       columns=["text"])
    X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), 
                        columns=["text"])
    
    y_true = X_test.sentiment
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

    return X_train, X_test, X_eval, y_true