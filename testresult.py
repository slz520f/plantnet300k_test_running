import pickle

with open('results/new_resnet18_model/new_resnet18_model.pkl', 'rb') as f:
    results = pickle.load(f)

# テスト結果を表示
print("Test Loss:", results['test_results']['loss'])
print("Test Accuracy:", results['test_results']['accuracy'])
print("Test Top-k Accuracy:", results['test_results']['topk_accuracy'])
