from sklearn import preprocessing

"""
- label encoding
- One hot encoding
- Binarization
"""



class CategoricalFeatures:
    def __init__ (self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names
        encoding_type: the type of encoding e.g. binary, one-hot, label
        """
        self.df = df
        self.categorical_features = categorical_features
        self.encoding_type = encoding_type
        self.output_df = self.df.copy(deep=True)

        if handle_na:
            for c in self.categorical_features:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna("-9999999")
    
    def _label_encoder(self):
        for c in self.categorical_features:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df


    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")
            
    def transform(self):


                
