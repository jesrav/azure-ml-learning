from model_class import FeatureSplitModel, LogRegWrapper

features = [
    "Pregnancies",
    "PlasmaGlucose",
    "DiastolicBloodPressure",
    "TricepsThickness",
    "SerumInsulin",
    "BMI",
    "DiabetesPedigree",
]

group_feature = "age_group"

feature_split_model_dict = {
    "over30": LogRegWrapper(
        features=features, params={"C": 100, "solver": "liblinear"}
    ),
    "under30": LogRegWrapper(
        features=features, params={"C": 10, "solver": "liblinear"}
    ),
}

model = FeatureSplitModel(
    features=features + ["age_group"],
    group_column="age_group",
    group_model_dict=feature_split_model_dict,
)
