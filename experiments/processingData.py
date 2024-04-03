import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


dataset_choice = "TVR"  # "TVR" or "QVH"
first_stage_retrieval_list = ["BM25", "PL2", "In_expB2"] if dataset_choice == "TVR" else ["BM25", "LemurTF_IDF", "In_expB2"]

feature_combinations_names = ["PL2 + DLH", "BM25 + Hiemstra_LM + PL2 + CoordinateMatch",
                              "BM25 (QE) + HiemstraLM (QE) + PL2 (QE) + CoordinateMatch (QE)",
                              "LemurTF_IDF + DirichletLM + DLH + Js_KLs"]

learned_models_names = ["linearSVR", "XGBoost (NDCG)", "FastRank Coordinate Ascent", "FastRank Random Forest"]

metrics = ["AP", "RR", "nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10", "nDCG@20", "nDCG@30", "nDCG@50", "R@1", "R@3", "R@5", "R@10", "R@20", "R@30", "R@50"]

'''
The data can be found in the "TVR" and "QVH" folders, each containing a "train" and "test" folder.
The format of the data is as follows:
    ,name,AP,RR,nDCG@1,nDCG@3,nDCG@5,nDCG@10,nDCG@20,nDCG@30,nDCG@50,R@1,R@3,R@5,R@10,R@20,R@30,R@50
    0,BM25_0_FastRank Coordinate Ascent,0.2621141556600122,0.2621141556600122,0.19971870604781997,0.24931210851918983,0.26710366025878696,0.28412105241418595,0.29958387125341196,0.3092160559872274,0.32052501724583204,0.19971870604781997,0.28270042194092826,0.3263009845288326,0.37834036568213786,0.44022503516174405,0.48523206751054854,0.5457102672292545
The format of the file names is as follows:
    <retrieval_method>_<fold_number>_<model_name>.csv
'''

# Load the data
def load_data(dataset_choice, data_type):
    data = {}
    for retrieval_method in first_stage_retrieval_list:
        data[retrieval_method] = {}
        for fold_number, feat_comb_loc in enumerate(feature_combinations_names):
            data[retrieval_method][feat_comb_loc] = {}
            for model_name in learned_models_names:
                file_name = f"{retrieval_method}_{fold_number}_{model_name}.csv"
                path = f"./{dataset_choice}/{data_type}/{file_name}"
                data[retrieval_method][feat_comb_loc][model_name] = pd.read_csv(path)
    return data


test_data = load_data(dataset_choice, "test")

# Create a heatmap with: x-axis = feature combinations, y-axis = learned models, color = RR
metric_choice = "RR"
first_stage_retrieval_choice = "In_expB2"

# create a 2d array with the values of the chosen metric
values = np.zeros((len(feature_combinations_names), len(learned_models_names)))
for feat_comb in feature_combinations_names:
    for model in learned_models_names:
        values[feature_combinations_names.index(feat_comb)][learned_models_names.index(model)] = test_data[first_stage_retrieval_choice][feat_comb][model].head()[metric_choice][0]

# create the heatmap
short_learned_models_names = ["SVR", "XGB", "FRC", "FRR"]
short_feature_combinations_names = ["Comb1", "Comb2", "Comb3", "Comb4"]
sns.heatmap(values, annot=True, xticklabels=short_learned_models_names, yticklabels=short_feature_combinations_names, cmap="Blues")
plt.xlabel("Learned Models")
plt.ylabel("Feature Combinations")
plt.title(f"RR values for {dataset_choice} using {first_stage_retrieval_choice}")
# save it to a file -> inside the plots folder
plt.savefig(f"./plots/{dataset_choice}_{first_stage_retrieval_choice}_{metric_choice}.png")
plt.show()

